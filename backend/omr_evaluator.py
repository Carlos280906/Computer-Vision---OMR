import os
import time
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)

# Import modul OMR utama
from omr_processor import OMRProcessor


# ═══════════════════════════════════════════════════════════
#  KONFIGURASI — SESUAIKAN DENGAN DATASET LO
# ═══════════════════════════════════════════════════════════

# Jumlah soal dan pilihan
NUM_QUESTIONS = 10   # Ganti sesuai lembar jawaban kalian
NUM_CHOICES   = 5    # A=0, B=1, C=2, D=3, E=4

# Folder dataset (relatif dari lokasi script ini)
DATASET_ROOT = Path("dataset")

# Output folder untuk grafik dan laporan
OUTPUT_DIR = Path("evaluation_results")

# ── GROUND TRUTH ────────────────────────────────────────────
# Format: { "nama_file.jpg": {1: 'A', 2: 'C', ...} }
# Isi ini secara manual sesuai jawaban asli tiap lembar.
#
# Untuk kunci jawaban (answer key), semua lembar dalam satu
# kondisi menggunakan kunci yang sama. Definisikan di bawah.

ANSWER_KEY = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E',
    6: 'A', 7: 'C', 8: 'B', 9: 'D', 10: 'A'
}

# Ground truth jawaban siswa per file
# Key = nama file (tanpa path), Value = dict {soal: jawaban}
GROUND_TRUTH = {
    # Contoh — ganti dengan data asli:
    "student_01.jpg": {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'A', 7:'C', 8:'B', 9:'D', 10:'A'},
    "student_02.jpg": {1:'B', 2:'B', 3:'A', 4:'D', 5:'C', 6:'E', 7:'C', 8:'B', 9:'A', 10:'A'},
    "student_03.jpg": {1:'A', 2:'C', 3:'C', 4:'B', 5:'E', 6:'A', 7:'D', 8:'B', 9:'D', 10:'E'},
}

# Kondisi pengujian yang akan dievaluasi
# Key = nama kondisi, Value = subfolder dalam DATASET_ROOT
CONDITIONS = {
    "Normal (Scan)"     : "normal",
    "Kamera HP"         : "camera",
    "Cahaya Kurang"     : "low_light",
    "Kertas Miring"     : "rotated",
    "Kertas Kotor/Noise": "noisy",
}

CHOICE_LABELS = ['A', 'B', 'C', 'D', 'E'][:NUM_CHOICES]


# ═══════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class SingleSheetMetrics:
    filename: str
    condition: str
    processing_time_ms: float
    detected: dict          # {soal: jawaban_terdeteksi}
    ground_truth: dict      # {soal: jawaban_benar}
    correct_bubbles: int
    total_bubbles: int
    accuracy: float
    error: Optional[str] = None


@dataclass
class ConditionSummary:
    condition: str
    num_sheets: int
    avg_accuracy: float
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_processing_time_ms: float
    sheet_results: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
#  EVALUATOR
# ═══════════════════════════════════════════════════════════

class OMREvaluator:

    def __init__(self):
        self.processor = OMRProcessor(
            num_questions=NUM_QUESTIONS,
            num_choices=NUM_CHOICES,
            bubble_fill_threshold=0.55,
            debug=False
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.all_results: list[SingleSheetMetrics] = []

    # ──────────────────────────────────────────────────────
    # MAIN ENTRY
    # ──────────────────────────────────────────────────────

    def run(self):
        print("\n" + "═"*60)
        print("  OMR EVALUATOR — AutoGrade System")
        print("  Binus University — Computer Vision Group 4")
        print("═"*60)

        condition_summaries = []

        for cond_name, cond_folder in CONDITIONS.items():
            folder = DATASET_ROOT / cond_folder
            if not folder.exists():
                print(f"\n[SKIP] Folder tidak ditemukan: {folder}")
                continue

            image_files = sorted([
                f for f in folder.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ])

            if not image_files:
                print(f"\n[SKIP] Tidak ada gambar di: {folder}")
                continue

            print(f"\n▶ Kondisi: {cond_name} ({len(image_files)} gambar)")
            print("-" * 50)

            sheet_results = []
            for img_path in image_files:
                result = self._evaluate_single(img_path, cond_name)
                sheet_results.append(result)
                self.all_results.append(result)

                status = "✓" if result.error is None else "✗"
                print(f"  {status} {img_path.name:30s} | "
                      f"Akurasi: {result.accuracy*100:5.1f}% | "
                      f"Waktu: {result.processing_time_ms:5.1f}ms")

            summary = self._summarize_condition(cond_name, sheet_results)
            condition_summaries.append(summary)

            print(f"\n  → Rata-rata akurasi : {summary.avg_accuracy*100:.2f}%")
            print(f"  → Rata-rata F1-Score: {summary.avg_f1:.4f}")
            print(f"  → Rata-rata waktu   : {summary.avg_processing_time_ms:.1f}ms")

        # ── Generate output ──
        print("\n" + "═"*60)
        print("  Generating reports & visualizations...")
        print("═"*60)

        self._plot_accuracy_per_condition(condition_summaries)
        self._plot_processing_time(condition_summaries)
        self._plot_confusion_matrix_all()
        self._plot_metrics_radar(condition_summaries)
        self._save_csv_report(condition_summaries)
        self._print_final_summary(condition_summaries)

        print(f"\n✅ Semua output tersimpan di: {OUTPUT_DIR.absolute()}")

    # ──────────────────────────────────────────────────────
    # EVALUATE SINGLE SHEET
    # ──────────────────────────────────────────────────────

    def _evaluate_single(self, img_path: Path, condition: str) -> SingleSheetMetrics:
        gt = GROUND_TRUTH.get(img_path.name, {})

        start = time.perf_counter()
        try:
            omr_result = self.processor.process(
                str(img_path), ANSWER_KEY, return_preview=False
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if omr_result.error:
                return SingleSheetMetrics(
                    filename=img_path.name, condition=condition,
                    processing_time_ms=elapsed_ms,
                    detected={}, ground_truth=gt,
                    correct_bubbles=0, total_bubbles=NUM_QUESTIONS,
                    accuracy=0.0, error=omr_result.error
                )

            # Rekonstruksi detected dict dari details
            detected = {
                d.question_number: d.detected_answer
                for d in omr_result.details
            }

            # Hitung akurasi bubble-level (vs ground truth, bukan kunci jawaban)
            correct = 0
            total = 0
            for q in range(1, NUM_QUESTIONS + 1):
                gt_ans = gt.get(q)
                det_ans = detected.get(q)
                if gt_ans is not None:
                    total += 1
                    if det_ans == gt_ans:
                        correct += 1

            accuracy = correct / total if total > 0 else 0.0

            return SingleSheetMetrics(
                filename=img_path.name, condition=condition,
                processing_time_ms=elapsed_ms,
                detected=detected, ground_truth=gt,
                correct_bubbles=correct, total_bubbles=total,
                accuracy=accuracy
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return SingleSheetMetrics(
                filename=img_path.name, condition=condition,
                processing_time_ms=elapsed_ms,
                detected={}, ground_truth=gt,
                correct_bubbles=0, total_bubbles=NUM_QUESTIONS,
                accuracy=0.0, error=str(e)
            )

    # ──────────────────────────────────────────────────────
    # SUMMARIZE CONDITION
    # ──────────────────────────────────────────────────────

    def _summarize_condition(self, cond_name, results) -> ConditionSummary:
        valid = [r for r in results if r.error is None and r.total_bubbles > 0]

        if not valid:
            return ConditionSummary(
                condition=cond_name, num_sheets=len(results),
                avg_accuracy=0, avg_precision=0, avg_recall=0,
                avg_f1=0, avg_processing_time_ms=0,
                sheet_results=results
            )

        # Kumpulkan semua prediksi vs ground truth untuk metrik agregat
        all_true, all_pred = [], []
        for r in valid:
            for q in range(1, NUM_QUESTIONS + 1):
                gt_ans = r.ground_truth.get(q)
                det_ans = r.detected.get(q)
                if gt_ans is not None:
                    all_true.append(gt_ans)
                    all_pred.append(det_ans if det_ans is not None else 'NONE')

        labels = CHOICE_LABELS
        precision = precision_score(all_true, all_pred, labels=labels, average='macro', zero_division=0)
        recall    = recall_score   (all_true, all_pred, labels=labels, average='macro', zero_division=0)
        f1        = f1_score       (all_true, all_pred, labels=labels, average='macro', zero_division=0)
        acc       = accuracy_score (all_true, all_pred)

        avg_time = np.mean([r.processing_time_ms for r in valid])

        return ConditionSummary(
            condition=cond_name,
            num_sheets=len(results),
            avg_accuracy=acc,
            avg_precision=precision,
            avg_recall=recall,
            avg_f1=f1,
            avg_processing_time_ms=avg_time,
            sheet_results=results
        )

    # ──────────────────────────────────────────────────────
    # VISUALIZATIONS
    # ──────────────────────────────────────────────────────

    def _plot_accuracy_per_condition(self, summaries: list[ConditionSummary]):
        """Grafik akurasi, precision, recall, F1 per kondisi."""
        labels   = [s.condition for s in summaries]
        acc_vals = [s.avg_accuracy * 100 for s in summaries]
        pre_vals = [s.avg_precision * 100 for s in summaries]
        rec_vals = [s.avg_recall * 100 for s in summaries]
        f1_vals  = [s.avg_f1 * 100 for s in summaries]

        x = np.arange(len(labels))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))
        bars_acc = ax.bar(x - 1.5*width, acc_vals, width, label='Accuracy',  color='#2196F3')
        bars_pre = ax.bar(x - 0.5*width, pre_vals, width, label='Precision', color='#4CAF50')
        bars_rec = ax.bar(x + 0.5*width, rec_vals, width, label='Recall',    color='#FF9800')
        bars_f1  = ax.bar(x + 1.5*width, f1_vals,  width, label='F1-Score',  color='#9C27B0')

        # Tambahkan value label
        for bars in [bars_acc, bars_pre, bars_rec, bars_f1]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Kondisi Pengujian', fontsize=11)
        ax.set_ylabel('Nilai Metrik (%)', fontsize=11)
        ax.set_title('Perbandingan Metrik Evaluasi per Kondisi Pengujian', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylim(0, 115)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.4, label='Target 90%')

        plt.tight_layout()
        out = OUTPUT_DIR / "fig1_metrics_per_condition.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] {out.name}")

    def _plot_processing_time(self, summaries: list[ConditionSummary]):
        """Grafik processing time per kondisi."""
        labels = [s.condition for s in summaries]
        times  = [s.avg_processing_time_ms for s in summaries]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(labels, times, color='#03A9F4', edgecolor='white')

        for bar, t in zip(bars, times):
            ax.text(t + 5, bar.get_y() + bar.get_height()/2,
                    f'{t:.1f} ms', va='center', fontsize=9)

        ax.set_xlabel('Rata-rata Waktu Pemrosesan (ms)', fontsize=11)
        ax.set_title('Processing Time per Kondisi Pengujian', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(times) * 1.25 if times else 100)

        plt.tight_layout()
        out = OUTPUT_DIR / "fig2_processing_time.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] {out.name}")

    def _plot_confusion_matrix_all(self):
        """Confusion matrix agregat dari semua data."""
        all_true, all_pred = [], []
        for r in self.all_results:
            if r.error is not None:
                continue
            for q in range(1, NUM_QUESTIONS + 1):
                gt_ans = r.ground_truth.get(q)
                det_ans = r.detected.get(q)
                if gt_ans is not None:
                    all_true.append(gt_ans)
                    all_pred.append(det_ans if det_ans is not None else 'NONE')

        if not all_true:
            print("  [SKIP] Tidak ada data untuk confusion matrix.")
            return

        # Sertakan 'NONE' sebagai label jika ada jawaban kosong
        labels = CHOICE_LABELS.copy()
        if 'NONE' in all_pred:
            labels.append('NONE')

        cm = confusion_matrix(all_true, all_pred, labels=labels)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Prediksi Sistem', fontsize=11)
        ax.set_ylabel('Ground Truth', fontsize=11)
        ax.set_title('Confusion Matrix — Semua Kondisi', fontsize=13, fontweight='bold')

        # Anotasi nilai
        thresh = cm.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center', fontsize=12,
                        color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        out = OUTPUT_DIR / "fig3_confusion_matrix.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] {out.name}")

    def _plot_metrics_radar(self, summaries: list[ConditionSummary]):
        """Radar chart perbandingan metrik antar kondisi."""
        if len(summaries) < 2:
            return

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

        for idx, s in enumerate(summaries):
            values = [
                s.avg_accuracy * 100,
                s.avg_precision * 100,
                s.avg_recall * 100,
                s.avg_f1 * 100
            ]
            values += values[:1]
            color = colors[idx % len(colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=s.condition, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
        ax.set_title('Radar Chart — Metrik per Kondisi', fontsize=13,
                     fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = OUTPUT_DIR / "fig4_radar_chart.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] {out.name}")

    # ──────────────────────────────────────────────────────
    # CSV REPORT
    # ──────────────────────────────────────────────────────

    def _save_csv_report(self, summaries: list[ConditionSummary]):
        """Simpan hasil ke CSV untuk tabel di paper."""

        # Summary per kondisi
        summary_path = OUTPUT_DIR / "tabel_summary_kondisi.csv"
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Kondisi', 'Jumlah Lembar',
                'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)',
                'Avg Processing Time (ms)'
            ])
            for s in summaries:
                writer.writerow([
                    s.condition,
                    s.num_sheets,
                    f"{s.avg_accuracy*100:.2f}",
                    f"{s.avg_precision*100:.2f}",
                    f"{s.avg_recall*100:.2f}",
                    f"{s.avg_f1*100:.2f}",
                    f"{s.avg_processing_time_ms:.1f}"
                ])
        print(f"  [Saved] {summary_path.name}")

        # Detail per lembar
        detail_path = OUTPUT_DIR / "tabel_detail_per_lembar.csv"
        with open(detail_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'File', 'Kondisi', 'Processing Time (ms)',
                'Correct Bubbles', 'Total Bubbles', 'Accuracy (%)', 'Error'
            ])
            for r in self.all_results:
                writer.writerow([
                    r.filename, r.condition,
                    f"{r.processing_time_ms:.1f}",
                    r.correct_bubbles, r.total_bubbles,
                    f"{r.accuracy*100:.2f}",
                    r.error or ''
                ])
        print(f"  [Saved] {detail_path.name}")

    # ──────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────

    def _print_final_summary(self, summaries: list[ConditionSummary]):
        print("\n" + "═"*60)
        print("  HASIL EVALUASI — RINGKASAN UNTUK BAB 4 PAPER")
        print("═"*60)

        header = f"{'Kondisi':<22} {'Acc%':>7} {'Prec%':>7} {'Rec%':>7} {'F1%':>7} {'Time(ms)':>10}"
        print(header)
        print("-"*60)

        for s in summaries:
            print(
                f"{s.condition:<22} "
                f"{s.avg_accuracy*100:>7.2f} "
                f"{s.avg_precision*100:>7.2f} "
                f"{s.avg_recall*100:>7.2f} "
                f"{s.avg_f1*100:>7.2f} "
                f"{s.avg_processing_time_ms:>10.1f}"
            )

        if summaries:
            all_acc = [s.avg_accuracy for s in summaries]
            print("-"*60)
            print(f"{'Overall Average':<22} {np.mean(all_acc)*100:>7.2f}")
            print(f"\n📄 File output tersimpan di folder: {OUTPUT_DIR}/")
            print("   ├── fig1_metrics_per_condition.png  → Grafik bar metrik")
            print("   ├── fig2_processing_time.png        → Grafik processing time")
            print("   ├── fig3_confusion_matrix.png       → Confusion matrix")
            print("   ├── fig4_radar_chart.png            → Radar chart")
            print("   ├── tabel_summary_kondisi.csv       → Tabel ringkasan")
            print("   └── tabel_detail_per_lembar.csv     → Detail tiap lembar")

        print("\n💡 Tips untuk Bab 4:")
        print("   • Masukkan fig1 & fig4 ke bagian Analysis")
        print("   • Masukkan fig3 (confusion matrix) ke bagian Discussion")
        print("   • Tabel CSV → copy-paste langsung ke Word/LaTeX")
        print("   • Bandingkan avg accuracy dengan paper referensi di proposal\n")


# ═══════════════════════════════════════════════════════════
#  QUICK TEST — tanpa dataset (generate dummy data)
# ═══════════════════════════════════════════════════════════

def run_quick_test():
    """
    Jalankan evaluasi dengan data simulasi (tanpa gambar asli).
    Berguna untuk verifikasi bahwa script berjalan dengan benar
    sebelum menggunakan dataset gambar nyata.
    """
    print("\n[QUICK TEST MODE] — Menggunakan data simulasi")
    print("Untuk evaluasi nyata, gunakan: evaluator.run()\n")

    np.random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Simulasi hasil deteksi
    summaries = []
    conditions_sim = {
        "Normal (Scan)"      : (0.97, 0.008),
        "Kamera HP"          : (0.91, 0.015),
        "Cahaya Kurang"      : (0.84, 0.020),
        "Kertas Miring"      : (0.88, 0.018),
        "Kertas Kotor/Noise" : (0.80, 0.025),
    }

    all_true_sim, all_pred_sim = [], []

    for cond_name, (base_acc, noise) in conditions_sim.items():
        acc   = min(1.0, max(0.0, base_acc + np.random.normal(0, noise)))
        prec  = min(1.0, max(0.0, acc + np.random.normal(0.01, 0.01)))
        rec   = min(1.0, max(0.0, acc + np.random.normal(-0.01, 0.01)))
        f1    = 2 * prec * rec / (prec + rec + 1e-6)
        t_ms  = np.random.uniform(80, 300)

        summaries.append(ConditionSummary(
            condition=cond_name, num_sheets=10,
            avg_accuracy=acc, avg_precision=prec,
            avg_recall=rec, avg_f1=f1,
            avg_processing_time_ms=t_ms
        ))

        # Simulasi confusion matrix data
        for _ in range(50):
            true_label = np.random.choice(CHOICE_LABELS)
            if np.random.random() < acc:
                pred_label = true_label
            else:
                pred_label = np.random.choice(CHOICE_LABELS)
            all_true_sim.append(true_label)
            all_pred_sim.append(pred_label)

    # Plot semua grafik dengan data simulasi
    evaluator = OMREvaluator.__new__(OMREvaluator)
    evaluator.all_results = []

    evaluator._plot_accuracy_per_condition(summaries)
    evaluator._plot_processing_time(summaries)

    # Confusion matrix dari data simulasi
    labels = CHOICE_LABELS
    cm = confusion_matrix(all_true_sim, all_pred_sim, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Prediksi Sistem', fontsize=11)
    ax.set_ylabel('Ground Truth', fontsize=11)
    ax.set_title('Confusion Matrix (Simulasi)', fontsize=13, fontweight='bold')
    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Saved] fig3_confusion_matrix.png")

    evaluator._plot_metrics_radar(summaries)
    evaluator._save_csv_report(summaries)
    evaluator._print_final_summary(summaries)


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv or not DATASET_ROOT.exists():
        # Mode simulasi: tidak butuh dataset asli
        run_quick_test()
    else:
        # Mode nyata: butuh folder dataset/ berisi gambar
        evaluator = OMREvaluator()
        evaluator.run()
