import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import base64
from itertools import combinations


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class BubbleResult:
    """Hasil deteksi satu soal"""
    question_number: int
    detected_answer: Optional[str]
    correct_answer: Optional[str]
    is_correct: bool
    confidence: float


@dataclass
class OMRResult:
    """Hasil keseluruhan penilaian satu lembar jawaban"""
    total_questions: int
    correct_count: int
    wrong_count: int
    empty_count: int
    score: float
    percentage: float
    details: list = field(default_factory=list)
    processed_image_b64: Optional[str] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────
# Core OMR Processor
# ─────────────────────────────────────────────

class OMRProcessor:
    """
    Proses lembar jawaban OMR menggunakan Computer Vision murni.

    PERBAIKAN v2:
    - Deteksi bubble berbasis kontur (bukan grid hardcode)
    - Support layout multi-kolom (5 blok soal horizontal)
    - Toleransi clustering Y/X yang lebih baik
    - Threshold pengisian bubble lebih toleran (0.38)
    - ROI bubble menggunakan ukuran bubble aktual dari kontur

    Parameters
    ----------
    num_questions : int
        Jumlah soal (default: 50)
    num_choices : int
        Jumlah pilihan per soal (default: 5 → A-E)
    num_column_blocks : int
        Jumlah blok kolom horizontal pada lembar jawaban.
        Contoh: lembar 50 soal biasanya 5 blok (tiap blok 10 soal).
        Set ke 1 untuk layout single-column. (default: 5)
    bubble_fill_threshold : float
        Ambang rasio isian bubble (0-1). Nilai lebih kecil = lebih sensitif.
        (default: 0.38)
    min_bubble_size_ratio : float
        Ukuran minimum bubble sebagai rasio dari dimensi terkecil gambar.
        (default: 0.015)
    max_bubble_size_ratio : float
        Ukuran maksimum bubble sebagai rasio dari dimensi terkecil gambar.
        (default: 0.12)
    debug : bool
        Jika True, cetak info debug ke stdout. (default: False)
    min_mark_density : float
        Minimum dark-pixel density inside a selected bubble.
    min_mark_coverage_ratio : float
        Minimum spread of dark cells across the bubble interior.
    """

    CHOICE_LABELS = ['A', 'B', 'C', 'D', 'E']

    def __init__(
        self,
        num_questions: int = 50,
        num_choices: int = 5,
        num_column_blocks: int = 5,
        bubble_fill_threshold: float = 0.38,
        min_bubble_size_ratio: float = 0.015,
        max_bubble_size_ratio: float = 0.12,
        debug: bool = False,
        min_mark_density: float = 0.35,
        min_mark_coverage_ratio: float = 0.45
    ):
        self.num_questions = num_questions
        self.num_choices = num_choices
        self.num_column_blocks = num_column_blocks
        self.choice_labels = self.CHOICE_LABELS[:num_choices]
        self.bubble_fill_threshold = bubble_fill_threshold
        self.min_bubble_size_ratio = min_bubble_size_ratio
        self.max_bubble_size_ratio = max_bubble_size_ratio
        self.debug = debug
        self.questions_per_block = num_questions // num_column_blocks
        self.min_mark_density = min_mark_density
        self.min_mark_coverage_ratio = min_mark_coverage_ratio

    # ──────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────

    def process(
        self,
        image_input,
        answer_key: dict,
        return_preview: bool = True
    ) -> OMRResult:
        """
        Proses satu lembar jawaban siswa dan hitung skor.

        Parameters
        ----------
        image_input : bytes | np.ndarray | str
            Raw image bytes, OpenCV array (BGR), atau path file.
        answer_key : dict
            Kunci jawaban. Format: {1: 'A', 2: 'C', ...}
        return_preview : bool
            Kembalikan gambar preview hasil (base64 PNG)?

        Returns
        -------
        OMRResult
        """
        try:
            image = self._load_image(image_input)
            gray, blurred, thresh = self._preprocess(image)
            warped, warped_thresh = self._detect_answer_sheet(image, gray, thresh)
            detected_answers, annotated = self._detect_bubbles(
                warped, warped_thresh, answer_key, return_preview
            )
            result = self._calculate_score(detected_answers, answer_key)
            if return_preview and annotated is not None:
                result.processed_image_b64 = self._encode_image(annotated)
            return result

        except SheetNotFoundError as e:
            return OMRResult(
                total_questions=self.num_questions,
                correct_count=0, wrong_count=0, empty_count=self.num_questions,
                score=0.0, percentage=0.0,
                error=f"Lembar jawaban tidak terdeteksi: {str(e)}"
            )
        except Exception as e:
            import traceback
            return OMRResult(
                total_questions=self.num_questions,
                correct_count=0, wrong_count=0, empty_count=self.num_questions,
                score=0.0, percentage=0.0,
                error=f"Error: {str(e)}\n{traceback.format_exc()}"
            )

    def process_answer_key_image(self, image_input) -> dict:
        """
        Baca kunci jawaban dari gambar kunci OMR.

        Returns
        -------
        dict : {1: 'A', 2: 'B', ...}
        """
        try:
            image = self._load_image(image_input)
            gray, blurred, thresh = self._preprocess(image)
            warped, warped_thresh = self._detect_answer_sheet(image, gray, thresh)
            answer_key, _ = self._detect_bubbles(warped, warped_thresh, {}, False)
            return answer_key
        except Exception as e:
            raise ValueError(f"Gagal membaca kunci jawaban: {str(e)}")

    # ──────────────────────────────────────────
    # STEP 1: LOAD IMAGE
    # ──────────────────────────────────────────

    def _load_image(self, image_input) -> np.ndarray:
        if isinstance(image_input, np.ndarray):
            return image_input
        elif isinstance(image_input, (bytes, bytearray)):
            arr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Tidak dapat membaca image bytes.")
            return img
        elif isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"File tidak ditemukan: {image_input}")
            return img
        else:
            raise TypeError(f"Tipe input tidak didukung: {type(image_input)}")

    # ──────────────────────────────────────────
    # STEP 2: PREPROCESSING
    # ──────────────────────────────────────────

    def _preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        # Resize ke lebar standar agar bubble punya ukuran yang konsisten
        if w > 1600:
            scale = 1600 / w
            image = cv2.resize(image, (1600, int(h * scale)))
        elif w < 800:
            scale = 800 / w
            image = cv2.resize(image, (800, int(h * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # GaussianBlur untuk kurangi noise kamera
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold: lebih robust terhadap variasi pencahayaan
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        return gray, blurred, thresh

    # ──────────────────────────────────────────
    # STEP 3: ANSWER REGION DETECTION
    # ──────────────────────────────────────────

    def _detect_answer_sheet(self, image, gray, thresh):
        """
        Deteksi area lembar jawaban dan lakukan perspective transform.
        Fallback ke seluruh gambar jika gagal.
        """
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        sheet_contour = None
        for c in contours[:10]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(c)
                img_area = image.shape[0] * image.shape[1]
                if area > img_area * 0.15:
                    sheet_contour = approx
                    break

        if sheet_contour is None:
            # Fallback: gunakan seluruh gambar
            h, w = image.shape[:2]
            sheet_contour = np.array([
                [[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]
            ], dtype=np.int32)

        warped = self._four_point_transform(image, sheet_contour.reshape(4, 2))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped_thresh = cv2.threshold(
            warped_gray, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        return warped, warped_thresh

    def _four_point_transform(self, image, pts):
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0], [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    # ──────────────────────────────────────────
    # STEP 4: BUBBLE DETECTION (DIPERBAIKI)
    # ──────────────────────────────────────────

    def _detect_bubbles(self, warped, warped_thresh, answer_key, annotate):
        """
        Deteksi bubble menggunakan pendekatan berbasis kontur dengan
        dukungan layout multi-kolom.

        Algoritma:
        1. Temukan semua kontur berbentuk lingkaran (kandidat bubble)
        2. Filter berdasarkan aspect ratio dan ukuran
        3. Cluster berdasarkan Y (baris soal) dengan toleransi yang cukup
        4. Skip baris header (biasanya punya terlalu sedikit bubble)
        5. Per baris: split bubble ke N blok kolom berdasarkan X
        6. Per blok: ambil 5 bubble terurut-X sebagai pilihan A-E
        7. Pilih bubble dengan intensitas (fill) tertinggi
        """
        h, w = warped.shape[:2]
        min_dim = min(w, h)

        # ── Temukan kandidat bubble ──
        contours, _ = cv2.findContours(
            warped_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bubble_contours = []
        for c in contours:
            (x, y, bw, bh) = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if area <= 0 or perimeter <= 0:
                continue

            ar = bw / float(bh)
            min_b = min(bw, bh)
            extent = area / float(bw * bh)
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

            # Filter aspect ratio dan ukuran relatif terhadap gambar
            if (0.70 <= ar <= 1.30) and \
               (min_dim * self.min_bubble_size_ratio <= min_b <= min_dim * self.max_bubble_size_ratio) and \
               (0.45 <= circularity <= 1.25) and \
               (extent >= 0.40):
                cx_center = x + bw // 2
                cy_center = y + bh // 2
                bubble_contours.append((cx_center, cy_center, bw, bh))

        bubble_contours = self._filter_by_dominant_bubble_size(bubble_contours)

        if self.debug:
            print(f"[DEBUG] Kandidat bubble: {len(bubble_contours)}")

        # ── Cluster bubble berdasarkan Y (baris) ──
        # Toleransi: 3% dari tinggi gambar (lebih besar dari versi lama 2%)
        row_tolerance = max(h * 0.03, 25)
        rows = self._cluster_bubbles_by_y(bubble_contours, row_tolerance)

        if self.debug:
            for i, row in enumerate(rows):
                ys = [b[1] for b in row]
                print(f"[DEBUG] Row {i+1}: {len(row)} bubbles, Y~{int(np.mean(ys))}")

        # ── Hilangkan baris header/label ──
        # Baris header biasanya punya jauh lebih sedikit bubble (hanya nomor section).
        # Gunakan threshold 80% dari total bubble yang diharapkan per baris.
        # Misal: 5 pilihan x 5 section = 25 bubble/baris → threshold = 20
        min_bubbles_per_row = int(self.num_choices * self.num_column_blocks * 0.8)
        question_rows = [r for r in rows if len(r) >= min_bubbles_per_row]

        if self.debug:
            print(f"[DEBUG] Question rows (setelah filter header): {len(question_rows)}")

        # Fallback jika kontur tidak cukup
        if len(question_rows) < self.questions_per_block:
            if self.debug:
                print("[DEBUG] Fallback ke grid-based detection")
            return self._grid_based_detection(warped, warped_thresh, answer_key, annotate)

        # Ambil sesuai jumlah baris per blok
        question_rows = question_rows[:self.questions_per_block]

        # ── Tentukan batas blok kolom berdasarkan distribusi X ──
        # Cari X dari semua bubble di question_rows, temukan gap besar
        all_cx = sorted([b[0] for row in question_rows for b in row])
        col_block_boundaries = self._find_column_boundaries(all_cx, self.num_column_blocks)

        if self.debug:
            print(f"[DEBUG] Column boundaries: {col_block_boundaries}")

        # ── Baca jawaban ──
        detected_answers = {}
        annotated = warped.copy() if annotate else None

        for row_i, row in enumerate(question_rows):
            row_sorted = sorted(row, key=lambda b: b[0])

            # Split ke blok kolom
            col_groups = [[] for _ in range(self.num_column_blocks)]
            for b in row_sorted:
                for si in range(self.num_column_blocks):
                    if col_block_boundaries[si] <= b[0] < col_block_boundaries[si + 1]:
                        col_groups[si].append(b)
                        break

            for sec_i, sec_bubbles in enumerate(col_groups):
                q_num = sec_i * self.questions_per_block + row_i + 1
                if q_num > self.num_questions:
                    continue

                # Pilih run A-E yang paling rapi secara geometri.
                # Ini mencegah kontur angka soal di kiri ikut dihitung sebagai pilihan.
                sec_sorted = self._select_choice_bubbles(sec_bubbles)

                if len(sec_sorted) < self.num_choices:
                    # Kurang dari num_choices bubble terdeteksi di blok ini
                    if self.debug:
                        print(f"[DEBUG] Q{q_num}: hanya {len(sec_sorted)} bubble terdeteksi")
                    detected_answers[q_num] = None
                    continue

                # Hitung isi tiap bubble. Printed A-E text should not count as a mark.
                fill_scores = []
                fill_densities = []
                coverage_ratios = []
                for b in sec_sorted:
                    cx, cy, bw, bh = b
                    score, density, coverage_ratio = self._measure_bubble_fill(
                        warped_thresh, cx, cy, bw, bh
                    )
                    fill_scores.append(score)
                    fill_densities.append(density)
                    coverage_ratios.append(coverage_ratio)

                max_val = max(fill_scores)
                max_idx = fill_scores.index(max_val)

                # Hitung fill ratio: max vs rata-rata sisanya
                other_vals = [v for i, v in enumerate(fill_scores) if i != max_idx]
                other_avg = float(np.mean(other_vals)) if other_vals else 0.0
                fill_ratio = max_val / (max_val + other_avg + 1e-6)

                # Pilih jawaban jika melewati threshold
                if (
                    fill_ratio >= self.bubble_fill_threshold and
                    fill_densities[max_idx] >= self.min_mark_density and
                    coverage_ratios[max_idx] >= self.min_mark_coverage_ratio
                ):
                    chosen = self.choice_labels[max_idx]
                    confidence = float(fill_ratio)
                else:
                    chosen = None
                    confidence = 0.0

                detected_answers[q_num] = chosen

                # ── Anotasi gambar preview ──
                if annotate and annotated is not None:
                    correct_ans = answer_key.get(q_num)
                    for i, b in enumerate(sec_sorted):
                        cx, cy, bw, bh = b
                        radius = min(bw, bh) // 2 + 3
                        if i == max_idx and chosen is not None:
                            if correct_ans and chosen == correct_ans:
                                color = (0, 200, 0)    # Hijau = benar
                            elif correct_ans:
                                color = (0, 0, 220)    # Merah = salah
                            else:
                                color = (200, 160, 0)  # Kuning = tanpa kunci
                            cv2.circle(annotated, (cx, cy), radius, color, 2)

                    # Label nomor soal
                    if sec_sorted:
                        x0 = sec_sorted[0][0]
                        y0 = sec_sorted[0][1]
                        cv2.putText(
                            annotated, str(q_num),
                            (max(0, x0 - 30), y0 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1
                        )

        return detected_answers, annotated

    # ──────────────────────────────────────────
    # HELPER: Cluster bubble by Y
    # ──────────────────────────────────────────

    def _filter_by_dominant_bubble_size(self, bubbles):
        """Keep contours close to the dominant printed bubble size."""
        if len(bubbles) < self.num_choices:
            return bubbles

        sizes = np.array([min(b[2], b[3]) for b in bubbles], dtype=np.float32)
        median_size = float(np.median(sizes))
        tolerance = max(4.0, median_size * 0.35)

        filtered = [
            b for b in bubbles
            if abs(min(b[2], b[3]) - median_size) <= tolerance
        ]

        return filtered if len(filtered) >= self.num_choices else bubbles

    def _select_choice_bubbles(self, sec_bubbles):
        """
        Return the A-E bubble run from one question block.

        If an accidental contour from the question number is present, simply
        taking the first five X positions shifts every answer. The real choices
        form a regular horizontal run, so choose the most even five-candidate
        group instead.
        """
        candidates = sorted(sec_bubbles, key=lambda b: b[0])
        if len(candidates) <= self.num_choices:
            return candidates if len(candidates) == self.num_choices else []

        index_groups = (
            combinations(range(len(candidates)), self.num_choices)
            if len(candidates) <= 12
            else (range(i, i + self.num_choices)
                  for i in range(0, len(candidates) - self.num_choices + 1))
        )

        best_group = None
        best_score = None
        all_x_span = candidates[-1][0] - candidates[0][0] + 1e-6

        for indexes in index_groups:
            group = [candidates[i] for i in indexes]
            xs = np.array([b[0] for b in group], dtype=np.float32)
            ys = np.array([b[1] for b in group], dtype=np.float32)
            sizes = np.array([min(b[2], b[3]) for b in group], dtype=np.float32)
            gaps = np.diff(xs)

            if len(gaps) == 0 or np.any(gaps <= 0):
                continue

            mean_gap = float(np.mean(gaps))
            median_size = float(np.median(sizes))
            if mean_gap < median_size * 1.05:
                continue

            gap_cv = float(np.std(gaps) / (mean_gap + 1e-6))
            size_cv = float(np.std(sizes) / (float(np.mean(sizes)) + 1e-6))
            y_span = float((np.max(ys) - np.min(ys)) / (median_size + 1e-6))

            if gap_cv > 0.55 or size_cv > 0.40 or y_span > 0.90:
                continue

            # Small preference for groups that start after a leading label
            # contour, while keeping geometry as the main signal.
            left_skip = float((group[0][0] - candidates[0][0]) / all_x_span)
            score = gap_cv * 4.0 + size_cv * 1.5 + y_span + max(0.0, 0.08 - left_skip)

            if best_score is None or score < best_score:
                best_score = score
                best_group = group

        if best_group is not None:
            return sorted(best_group, key=lambda b: b[0])

        return candidates[-self.num_choices:]

    def _measure_bubble_fill(self, thresh_img, cx, cy, bw, bh):
        """
        Measure a real mark inside a bubble, not just printed A-E text.

        Returns (score, density, coverage_ratio). Printed letters can create
        dark pixels, but they rarely cover the inner circle evenly.
        """
        h, w = thresh_img.shape[:2]
        outer_radius = max(1, min(bw, bh) // 2)
        inner_radius = max(1, int(outer_radius * 0.72))

        x1 = max(0, cx - inner_radius)
        x2 = min(w, cx + inner_radius + 1)
        y1 = max(0, cy - inner_radius)
        y2 = min(h, cy + inner_radius + 1)

        roi = thresh_img[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0, 0.0, 0.0

        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx - x1, cy - y1), inner_radius, 255, -1)
        mask_area = cv2.countNonZero(mask)
        if mask_area == 0:
            return 0.0, 0.0, 0.0

        marked = cv2.bitwise_and(roi, roi, mask=mask)
        density = cv2.countNonZero(marked) / float(mask_area)

        grid_size = 5
        solid_cells = 0
        valid_cells = 0
        min_cell_area = max(3, mask_area / float(grid_size * grid_size) * 0.35)

        for gy in range(grid_size):
            y_start = int(round(gy * roi.shape[0] / grid_size))
            y_end = int(round((gy + 1) * roi.shape[0] / grid_size))
            for gx in range(grid_size):
                x_start = int(round(gx * roi.shape[1] / grid_size))
                x_end = int(round((gx + 1) * roi.shape[1] / grid_size))

                cell_mask = mask[y_start:y_end, x_start:x_end]
                cell_marked = marked[y_start:y_end, x_start:x_end]
                cell_area = cv2.countNonZero(cell_mask)
                if cell_area < min_cell_area:
                    continue

                valid_cells += 1
                cell_density = cv2.countNonZero(cell_marked) / float(cell_area)
                if cell_density >= 0.35:
                    solid_cells += 1

        coverage_ratio = solid_cells / float(valid_cells) if valid_cells else 0.0
        score = (density * 0.55) + (coverage_ratio * 0.45)
        return float(score), float(density), float(coverage_ratio)

    def _cluster_bubbles_by_y(self, bubbles, tolerance):
        """Kelompokkan bubble berdasarkan posisi Y dengan toleransi."""
        if not bubbles:
            return []
        bubbles_sorted = sorted(bubbles, key=lambda b: b[1])
        rows = []
        current = [bubbles_sorted[0]]
        for b in bubbles_sorted[1:]:
            # Bandingkan dengan rata-rata Y cluster saat ini
            avg_y = np.mean([bb[1] for bb in current])
            if abs(b[1] - avg_y) <= tolerance:
                current.append(b)
            else:
                rows.append(current)
                current = [b]
        rows.append(current)
        return rows

    # ──────────────────────────────────────────
    # HELPER: Find column block boundaries
    # ──────────────────────────────────────────

    def _find_column_boundaries(self, all_cx_sorted, num_blocks):
        """
        Temukan batas blok kolom berdasarkan gap terbesar pada distribusi X.

        Strategi:
        1. Kelompokkan X yang berdekatan (±35px) menjadi col_group
        2. Cari (num_blocks-1) gap terbesar ANTAR col_group
        3. Gunakan titik tengah gap sebagai batas blok

        Ini lebih akurat daripada mencari gap pada raw X, karena
        gap kecil antar bubble dalam satu section tidak mengganggu.

        Returns list of (num_blocks+1) boundary values:
        [0, separator1, separator2, ..., max_x+100]
        """
        if len(all_cx_sorted) < 2:
            w_approx = all_cx_sorted[-1] * 2 if all_cx_sorted else 1000
            step = w_approx // num_blocks
            return [i * step for i in range(num_blocks + 1)]

        # Step 1: Cluster X menjadi col_group centers
        col_group_centers = []
        cur_group = [all_cx_sorted[0]]
        for x in all_cx_sorted[1:]:
            if x - cur_group[-1] <= 35:
                cur_group.append(x)
            else:
                col_group_centers.append(int(np.mean(cur_group)))
                cur_group = [x]
        col_group_centers.append(int(np.mean(cur_group)))

        if len(col_group_centers) < 2:
            # Fallback jika terlalu sedikit group
            step = (all_cx_sorted[-1] - all_cx_sorted[0]) // num_blocks
            start = all_cx_sorted[0]
            return [0] + [start + i * step for i in range(1, num_blocks)] + [all_cx_sorted[-1] + 100]

        # Step 2: Hitung gap antar col_group
        gaps = []
        for i in range(1, len(col_group_centers)):
            gap = col_group_centers[i] - col_group_centers[i - 1]
            gaps.append((gap, col_group_centers[i - 1], col_group_centers[i]))

        # Step 3: Ambil (num_blocks-1) gap terbesar
        gaps_sorted = sorted(gaps, reverse=True)
        separator_xs = sorted([
            g[1] + (g[2] - g[1]) // 2
            for g in gaps_sorted[:num_blocks - 1]
        ])

        boundaries = [0] + separator_xs + [all_cx_sorted[-1] + 100]
        return boundaries

    # ──────────────────────────────────────────
    # FALLBACK: Grid-based detection
    # ──────────────────────────────────────────

    def _grid_based_detection(self, warped, warped_thresh, answer_key, annotate):
        """
        Fallback berbasis sampling grid uniform.
        Digunakan jika kontur-based detection gagal.
        Dibagi ke num_column_blocks blok horizontal.
        """
        h, w = warped.shape[:2]

        margin_top = int(h * 0.12)
        margin_bottom = int(h * 0.05)
        margin_left = int(w * 0.05)
        margin_right = int(w * 0.05)

        grid_h = h - margin_top - margin_bottom
        grid_w = w - margin_left - margin_right

        # Tiap blok punya lebar grid_w / num_column_blocks
        block_w = grid_w / self.num_column_blocks
        row_step = grid_h / self.questions_per_block
        col_step = block_w / self.num_choices
        bubble_r = int(min(row_step, col_step) * 0.35)

        detected_answers = {}
        annotated = warped.copy() if annotate else None

        for q in range(self.num_questions):
            question_num = q + 1
            sec_i = (q) // self.questions_per_block
            row_i = (q) % self.questions_per_block

            cy_center = int(margin_top + (row_i + 0.5) * row_step)
            block_start_x = margin_left + sec_i * int(block_w)

            intensities = []
            fill_densities = []
            coverage_ratios = []
            for c in range(self.num_choices):
                cx_center = int(block_start_x + (c + 0.5) * col_step)
                score, density, coverage_ratio = self._measure_bubble_fill(
                    warped_thresh,
                    cx_center,
                    cy_center,
                    bubble_r * 2,
                    bubble_r * 2
                )
                intensities.append(score)
                fill_densities.append(density)
                coverage_ratios.append(coverage_ratio)

            if max(intensities) <= 0:
                detected_answers[question_num] = None
                continue

            max_idx = intensities.index(max(intensities))
            other_avg = np.mean([v for i, v in enumerate(intensities) if i != max_idx]) \
                if len(intensities) > 1 else 0
            fill_ratio = intensities[max_idx] / (intensities[max_idx] + other_avg + 1e-6)

            detected_answers[question_num] = (
                self.choice_labels[max_idx]
                if (
                    fill_ratio >= self.bubble_fill_threshold and
                    fill_densities[max_idx] >= self.min_mark_density and
                    coverage_ratios[max_idx] >= self.min_mark_coverage_ratio
                )
                else None
            )

        return detected_answers, annotated

    # ──────────────────────────────────────────
    # STEP 5: SCORING
    # ──────────────────────────────────────────

    def _calculate_score(self, detected: dict, answer_key: dict) -> OMRResult:
        correct = 0
        wrong = 0
        empty = 0
        details = []

        for q_num in range(1, self.num_questions + 1):
            detected_ans = detected.get(q_num)
            correct_ans = answer_key.get(q_num)

            if detected_ans is None:
                empty += 1
                is_correct = False
                confidence = 0.0
            elif correct_ans is None:
                wrong += 1
                is_correct = False
                confidence = 1.0
            elif detected_ans == correct_ans:
                correct += 1
                is_correct = True
                confidence = 1.0
            else:
                wrong += 1
                is_correct = False
                confidence = 1.0

            details.append(BubbleResult(
                question_number=q_num,
                detected_answer=detected_ans,
                correct_answer=correct_ans,
                is_correct=is_correct,
                confidence=confidence
            ))

        total_with_key = sum(1 for q in range(1, self.num_questions + 1) if answer_key.get(q))
        if total_with_key == 0:
            total_with_key = self.num_questions

        score = (correct / total_with_key) * 100

        return OMRResult(
            total_questions=self.num_questions,
            correct_count=correct,
            wrong_count=wrong,
            empty_count=empty,
            score=round(score, 2),
            percentage=round(score, 2),
            details=details
        )

    # ──────────────────────────────────────────
    # HELPER: Encode image
    # ──────────────────────────────────────────

    def _encode_image(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')


# ─────────────────────────────────────────────
# Custom Exception
# ─────────────────────────────────────────────

class SheetNotFoundError(Exception):
    """Raised saat lembar jawaban tidak dapat dideteksi dalam gambar."""
    pass


# ─────────────────────────────────────────────
# Quick Test (jalankan langsung: python omr_processor.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python omr_processor.py <image_path> [answer_key_image_path]")
        print("       python omr_processor.py <image_path> --key A,B,C,D,...")
        sys.exit(1)

    processor = OMRProcessor(
        num_questions=50,
        num_choices=5,
        num_column_blocks=5,
        bubble_fill_threshold=0.38,
        debug=True
    )

    image_path = sys.argv[1]

    # Baca kunci jawaban
    answer_key = {}
    if len(sys.argv) >= 4 and sys.argv[2] == "--key":
        labels_str = sys.argv[3].upper().split(',')
        for i, label in enumerate(labels_str):
            label = label.strip()
            if label:
                answer_key[i + 1] = label
        print(f"Kunci jawaban dari argumen: {answer_key}")
    elif len(sys.argv) >= 3 and sys.argv[2] != "--key":
        key_image_path = sys.argv[2]
        print(f"Membaca kunci jawaban dari: {key_image_path}")
        answer_key = processor.process_answer_key_image(key_image_path)
        print(f"Kunci jawaban terdeteksi: {answer_key}")

    # Proses lembar jawaban
    print(f"\nMemproses: {image_path}")
    result = processor.process(image_path, answer_key, return_preview=False)

    if result.error:
        print(f"\nERROR: {result.error}")
    else:
        print(f"\n{'='*40}")
        print(f"Total Soal : {result.total_questions}")
        print(f"Benar      : {result.correct_count}")
        print(f"Salah      : {result.wrong_count}")
        print(f"Kosong     : {result.empty_count}")
        print(f"Skor       : {result.score:.2f}")
        print(f"{'='*40}")
        print("\nDetail per soal:")
        for d in result.details:
            status = "✓" if d.is_correct else ("○" if d.detected_answer is None else "✗")
            print(f"  Q{d.question_number:2d}: Jawab={d.detected_answer or '-':>1}  "
                  f"Kunci={d.correct_answer or '-':>1}  {status}")
