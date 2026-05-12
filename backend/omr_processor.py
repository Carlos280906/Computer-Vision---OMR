import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import base64
import io


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class BubbleResult:
    """Hasil deteksi satu soal"""
    question_number: int          # Nomor soal (1-based)
    detected_answer: Optional[str]  # Jawaban terdeteksi ('A','B','C','D','E' atau None)
    correct_answer: Optional[str]  # Jawaban kunci
    is_correct: bool              # Benar/salah
    confidence: float             # Skor kepercayaan deteksi (0.0 - 1.0)


@dataclass
class OMRResult:
    """Hasil keseluruhan penilaian satu lembar jawaban"""
    total_questions: int
    correct_count: int
    wrong_count: int
    empty_count: int
    score: float                  # Skor 0-100
    percentage: float             # Persentase kebenaran
    details: list = field(default_factory=list)  # List[BubbleResult]
    processed_image_b64: Optional[str] = None    # Preview hasil (base64 PNG)
    error: Optional[str] = None


# ─────────────────────────────────────────────
# Core OMR Processor
# ─────────────────────────────────────────────

class OMRProcessor:
    """
    Proses lembar jawaban OMR menggunakan Computer Vision murni.
    Tidak ada model ML, tidak perlu dataset atau training.

    Parameters
    ----------
    num_questions : int
        Jumlah soal dalam lembar jawaban (default: 50)
    num_choices : int
        Jumlah pilihan per soal, misal 4 = A-D, 5 = A-E (default: 5)
    choice_labels : list[str]
        Label pilihan jawaban (default: ['A','B','C','D','E'])
    bubble_fill_threshold : float
        Ambang batas (0-1) untuk menentukan bubble "diisi".
        Nilai lebih kecil = lebih sensitif (default: 0.5)
    debug : bool
        Jika True, simpan gambar debug ke disk (default: False)
    """

    CHOICE_LABELS = ['A', 'B', 'C', 'D', 'E']

    def __init__(
        self,
        num_questions: int = 50,
        num_choices: int = 5,
        choice_labels: Optional[list] = None,
        bubble_fill_threshold: float = 0.5,
        debug: bool = False
    ):
        self.num_questions = num_questions
        self.num_choices = num_choices
        self.choice_labels = choice_labels or self.CHOICE_LABELS[:num_choices]
        self.bubble_fill_threshold = bubble_fill_threshold
        self.debug = debug

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
            - bytes  : raw image bytes (dari upload API)
            - ndarray: gambar OpenCV (BGR)
            - str    : path file gambar
        answer_key : dict
            Kunci jawaban. Format: {1: 'A', 2: 'C', 3: 'B', ...}
            Key = nomor soal (int), Value = jawaban ('A'-'E')
        return_preview : bool
            Apakah kembalikan gambar preview hasil (base64)?

        Returns
        -------
        OMRResult
        """
        try:
            # 1. Load gambar
            image = self._load_image(image_input)

            # 2. Preprocessing
            gray, blurred, thresh = self._preprocess(image)

            # 3. Deteksi area lembar jawaban (perspective correction)
            warped, warped_thresh = self._detect_answer_sheet(image, gray, thresh)

            # 4. Deteksi bubble dan klasifikasi jawaban
            detected_answers, annotated = self._detect_bubbles(
                warped, warped_thresh, answer_key, return_preview
            )

            # 5. Scoring
            result = self._calculate_score(detected_answers, answer_key)

            # 6. Preview image
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
            return OMRResult(
                total_questions=self.num_questions,
                correct_count=0, wrong_count=0, empty_count=self.num_questions,
                score=0.0, percentage=0.0,
                error=f"Error processing image: {str(e)}"
            )

    def process_answer_key_image(self, image_input) -> dict:
        """
        Baca kunci jawaban dari gambar kunci (format sama dengan lembar siswa).

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
        """Load gambar dari berbagai format input."""
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
        """
        Konversi ke grayscale, gaussian blur, dan adaptive threshold.
        Teknik ini umum digunakan dalam sistem OMR untuk mempersiapkan
        gambar sebelum deteksi kontur [ref: proposal Bab Methodology].
        """
        # Resize ke lebar standar agar konsisten
        h, w = image.shape[:2]
        if w > 1200:
            scale = 1200 / w
            image = cv2.resize(image, (1200, int(h * scale)))

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur → kurangi noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive Threshold → binarisasi gambar
        # Lebih robust terhadap variasi pencahayaan dibanding global threshold
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
        Deteksi area lembar jawaban menggunakan kontur dan
        lakukan perspective transform (koreksi sudut/kemiringan).

        Ini mengatasi masalah misalignment yang umum terjadi
        saat gambar diambil menggunakan kamera [ref: proposal].
        """
        # Temukan semua kontur
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Urutkan dari terbesar
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        sheet_contour = None
        for c in contours[:10]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(c)
                img_area = image.shape[0] * image.shape[1]
                # Minimal 20% dari luas gambar
                if area > img_area * 0.2:
                    sheet_contour = approx
                    break

        if sheet_contour is None:
            # Fallback: gunakan seluruh gambar
            h, w = image.shape[:2]
            sheet_contour = np.array([
                [[0, 0]], [[w-1, 0]], [[w-1, h-1]], [[0, h-1]]
            ], dtype=np.int32)

        # Perspective transform
        warped = self._four_point_transform(image, sheet_contour.reshape(4, 2))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped_thresh = cv2.threshold(
            warped_gray, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        return warped, warped_thresh

    def _four_point_transform(self, image, pts):
        """
        Perspective transformation 4-titik untuk meluruskan gambar.
        Teknik homography — sama seperti yang digunakan pada aplikasi
        mobile OMR [ref: Largo et al., 2022 dalam proposal].
        """
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def _order_points(self, pts):
        """Urutkan 4 titik: TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # TL: x+y terkecil
        rect[2] = pts[np.argmax(s)]   # BR: x+y terbesar
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR: y-x terkecil
        rect[3] = pts[np.argmax(diff)]  # BL: y-x terbesar
        return rect

    # ──────────────────────────────────────────
    # STEP 4 & 5: BUBBLE DETECTION + CLASSIFICATION
    # ──────────────────────────────────────────

    def _detect_bubbles(self, warped, warped_thresh, answer_key, annotate):
        """
        Deteksi bubble yang diarsir menggunakan analisis intensitas pixel.

        Pendekatan:
        - Temukan semua kontur yang berbentuk lingkaran (bubble)
        - Grid-kan posisi bubble berdasarkan baris (soal) & kolom (pilihan)
        - Hitung rata-rata pixel putih per bubble → yang tertinggi = pilihan jawaban
        
        Ini merupakan pendekatan yang digunakan oleh Atencio et al. (2023)
        dan Palak et al. (2025) dalam proposal [ref: proposal Related Work].
        """
        h, w = warped.shape[:2]

        # ── Temukan semua kontur lingkaran (kandidat bubble) ──
        contours, _ = cv2.findContours(
            warped_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bubble_contours = []
        for c in contours:
            (x, y, bw, bh) = cv2.boundingRect(c)
            ar = bw / float(bh)  # Aspect ratio

            # Filter: bentuk hampir persegi/lingkaran, ukuran masuk akal
            min_dim = min(w, h) * 0.01   # Minimal 1% dari dimensi terkecil
            max_dim = min(w, h) * 0.08   # Maksimal 8%

            if (0.7 <= ar <= 1.3) and (min_dim <= bw <= max_dim):
                bubble_contours.append(c)

        if len(bubble_contours) < self.num_questions * self.num_choices:
            # Fallback: gunakan grid sampling jika kontur tidak cukup
            return self._grid_based_detection(warped, warped_thresh, answer_key, annotate)

        # ── Grid-kan bubble berdasarkan posisi Y (baris = soal) ──
        bounding_boxes = [cv2.boundingRect(c) for c in bubble_contours]
        
        # Kelompokkan berdasarkan baris (Y)
        row_tolerance = h * 0.02  # 2% toleransi untuk dianggap satu baris
        rows = self._cluster_by_position(
            bounding_boxes, axis=1, tolerance=row_tolerance
        )
        rows = rows[:self.num_questions]  # Ambil sesuai jumlah soal

        # Kelompokkan berdasarkan kolom (X)
        col_tolerance = w * 0.02
        cols_global = self._cluster_by_position(
            bounding_boxes, axis=0, tolerance=col_tolerance
        )

        # ── Klasifikasi per soal ──
        detected_answers = {}
        annotated = warped.copy() if annotate else None

        for q_idx, row_boxes in enumerate(rows):
            question_num = q_idx + 1

            # Urutkan bubble dalam baris berdasarkan X
            row_sorted = sorted(row_boxes, key=lambda b: b[0])
            row_sorted = row_sorted[:self.num_choices]

            intensities = []
            for (x, y, bw, bh) in row_sorted:
                # Hitung rata-rata pixel putih dalam ROI bubble
                roi = warped_thresh[y:y+bh, x:x+bw]
                mean_val = cv2.mean(roi)[0]
                intensities.append(mean_val)

            if not intensities:
                detected_answers[question_num] = None
                continue

            max_intensity = max(intensities)
            max_idx = intensities.index(max_intensity)

            # Tentukan apakah bubble benar-benar diisi
            # (bandingkan vs rata-rata intensitas bubble lain)
            other_avg = (
                np.mean([v for i, v in enumerate(intensities) if i != max_idx])
                if len(intensities) > 1 else 0
            )
            fill_ratio = max_intensity / (max_intensity + other_avg + 1e-6)

            if fill_ratio >= self.bubble_fill_threshold:
                chosen = self.choice_labels[max_idx] if max_idx < len(self.choice_labels) else None
                confidence = float(fill_ratio)
            else:
                chosen = None
                confidence = 0.0

            detected_answers[question_num] = chosen

            # Anotasi gambar preview
            if annotate and annotated is not None:
                correct_ans = answer_key.get(question_num)
                for i, (x, y, bw, bh) in enumerate(row_sorted):
                    cx, cy = x + bw // 2, y + bh // 2
                    if i == max_idx and chosen is not None:
                        if correct_ans and chosen == correct_ans:
                            color = (0, 200, 0)    # Hijau = benar
                            thickness = 2
                        elif correct_ans:
                            color = (0, 0, 220)    # Merah = salah
                            thickness = 2
                        else:
                            color = (200, 200, 0)  # Kuning = tidak ada kunci
                            thickness = 2
                        cv2.circle(annotated, (cx, cy), max(bw, bh)//2 + 2, color, thickness)
                    # Nomor soal
                if len(row_sorted) > 0:
                    x0, y0, _, _ = row_sorted[0]
                    cv2.putText(
                        annotated, str(question_num),
                        (max(0, x0 - 25), y0 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1
                    )

        return detected_answers, annotated

    def _grid_based_detection(self, warped, warped_thresh, answer_key, annotate):
        """
        Fallback: sampling berbasis grid uniform.
        Digunakan jika kontur-based detection gagal menemukan cukup bubble.
        Cocok untuk lembar jawaban dengan format standar yang sudah diketahui.
        """
        h, w = warped.shape[:2]

        # Estimasi area bubble grid (asumsikan bubble ada di 80% tengah gambar)
        margin_top = int(h * 0.10)
        margin_bottom = int(h * 0.05)
        margin_left = int(w * 0.10)
        margin_right = int(w * 0.05)

        grid_h = h - margin_top - margin_bottom
        grid_w = w - margin_left - margin_right

        row_step = grid_h / self.num_questions
        col_step = grid_w / self.num_choices

        bubble_r = int(min(row_step, col_step) * 0.35)

        detected_answers = {}
        annotated = warped.copy() if annotate else None

        for q in range(self.num_questions):
            question_num = q + 1
            cy_center = int(margin_top + (q + 0.5) * row_step)
            intensities = []

            for c in range(self.num_choices):
                cx_center = int(margin_left + (c + 0.5) * col_step)

                # Sampling ROI
                y1 = max(0, cy_center - bubble_r)
                y2 = min(h, cy_center + bubble_r)
                x1 = max(0, cx_center - bubble_r)
                x2 = min(w, cx_center + bubble_r)

                roi = warped_thresh[y1:y2, x1:x2]
                intensities.append(float(cv2.mean(roi)[0]))

            if max(intensities) < 20:  # Tidak ada yang diisi
                detected_answers[question_num] = None
                continue

            max_idx = intensities.index(max(intensities))
            other_avg = np.mean([v for i, v in enumerate(intensities) if i != max_idx]) if len(intensities) > 1 else 0
            fill_ratio = intensities[max_idx] / (intensities[max_idx] + other_avg + 1e-6)

            if fill_ratio >= self.bubble_fill_threshold:
                chosen = self.choice_labels[max_idx]
            else:
                chosen = None

            detected_answers[question_num] = chosen

        return detected_answers, annotated

    # ──────────────────────────────────────────
    # STEP 6: SCORING
    # ──────────────────────────────────────────

    def _calculate_score(self, detected: dict, answer_key: dict) -> OMRResult:
        """
        Bandingkan jawaban siswa dengan kunci jawaban dan hitung skor.
        Setiap jawaban benar mendapat poin proporsional (100 / jumlah soal).
        """
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
                # Kunci tidak tersedia untuk soal ini
                is_correct = False
                confidence = 1.0
                wrong += 1
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
        percentage = score

        return OMRResult(
            total_questions=self.num_questions,
            correct_count=correct,
            wrong_count=wrong,
            empty_count=empty,
            score=round(score, 2),
            percentage=round(percentage, 2),
            details=details
        )

    # ──────────────────────────────────────────
    # HELPER
    # ──────────────────────────────────────────

    def _cluster_by_position(self, bboxes, axis, tolerance):
        """Kelompokkan bounding boxes berdasarkan posisi (X atau Y) dengan toleransi."""
        coords = [b[axis] for b in bboxes]
        sorted_idx = np.argsort(coords)
        clusters = []
        current = [bboxes[sorted_idx[0]]]

        for i in sorted_idx[1:]:
            if abs(bboxes[i][axis] - current[-1][axis]) <= tolerance:
                current.append(bboxes[i])
            else:
                clusters.append(current)
                current = [bboxes[i]]
        clusters.append(current)
        return clusters

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode gambar OpenCV ke base64 PNG string untuk dikirim via API."""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')


# ─────────────────────────────────────────────
# Custom Exception
# ─────────────────────────────────────────────

class SheetNotFoundError(Exception):
    """Raised saat lembar jawaban tidak dapat dideteksi dalam gambar."""
    pass

