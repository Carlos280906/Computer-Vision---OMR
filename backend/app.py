from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback
 
from omr_processor import OMRProcessor
 
 
# ═══════════════════════════════════════════════════════════
#  INISIALISASI
# ═══════════════════════════════════════════════════════════
 
app = FastAPI(
    title="AutoGrade OMR API",
    version="1.0.0"
)
 
# CORS — izinkan frontend Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Inisialisasi OMR Processor
processor = OMRProcessor(
    num_questions=50,
    num_choices=5,
    bubble_fill_threshold=0.55,
)
 
 
# ═══════════════════════════════════════════════════════════
#  RESPONSE MODELS
# ═══════════════════════════════════════════════════════════
 
class GradeResponse(BaseModel):
    success: bool
    score: float                        # 0-100
    status: str                         # "Lulus" / "Tidak Lulus"
    correct_count: int
    wrong_count: int
    empty_count: int
    total_questions: int
    details: list                       # Per soal: [{question, detected, correct, is_correct}]
    preview_image: Optional[str] = None # Base64 PNG preview hasil koreksi
    error: Optional[str] = None
 
class HealthResponse(BaseModel):
    status: str
    message: str
 
 
# ═══════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════
 
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Cek apakah backend berjalan — bisa dipanggil frontend saat pertama load."""
    return {"status": "ok", "message": "AutoGrade OMR Backend is running"}
 
 
@app.post("/api/grade", response_model=GradeResponse)
async def grade(
    answer_key_image: UploadFile = File(..., description="Gambar kunci jawaban"),
    student_sheet:    UploadFile = File(..., description="Gambar lembar jawaban siswa"),
    passing_score:    float      = 60.0,
):
    """
    Endpoint utama — dipanggil tombol 'Proses Penilaian' di frontend.
 
    Menerima:
      - answer_key_image : file gambar kunci jawaban  (file1 di frontend)
      - student_sheet    : file gambar lembar siswa   (file2 di frontend)
      - passing_score    : nilai minimum lulus (default 60)
 
    Mengembalikan:
      - score, status Lulus/Tidak Lulus, detail per soal, preview gambar
    """
    ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"]
    if answer_key_image.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Format kunci jawaban tidak didukung: {answer_key_image.content_type}")
    if student_sheet.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Format lembar siswa tidak didukung: {student_sheet.content_type}")
 
    try:
        # Baca bytes dari upload
        key_bytes     = await answer_key_image.read()
        student_bytes = await student_sheet.read()
 
        # ── Step 1: Baca kunci jawaban dari gambar ──
        try:
            answer_key = processor.process_answer_key_image(key_bytes)
        except Exception as e:
            raise HTTPException(422, f"Gagal membaca kunci jawaban: {str(e)}")
 
        if not answer_key:
            raise HTTPException(422, "Kunci jawaban tidak terdeteksi. Pastikan gambar jelas.")
 
        # ── Step 2: Proses lembar jawaban siswa ──
        result = processor.process(
            image_input=student_bytes,
            answer_key=answer_key,
            return_preview=True
        )
 
        if result.error:
            return GradeResponse(
                success=False,
                score=0,
                status="Error",
                correct_count=0,
                wrong_count=0,
                empty_count=result.total_questions,
                total_questions=result.total_questions,
                details=[],
                error=result.error
            )
 
        # ── Step 3: Tentukan status lulus/tidak ──
        status = "Lulus" if result.score >= passing_score else "Tidak Lulus"
 
        # ── Step 4: Format detail per soal untuk frontend ──
        details = [
            {
                "question":   d.question_number,
                "detected":   d.detected_answer,
                "correct":    d.correct_answer,
                "is_correct": d.is_correct,
            }
            for d in result.details
        ]
 
        return GradeResponse(
            success=True,
            score=result.score,
            status=status,
            correct_count=result.correct_count,
            wrong_count=result.wrong_count,
            empty_count=result.empty_count,
            total_questions=result.total_questions,
            details=details,
            preview_image=result.processed_image_b64,
        )
 
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Internal server error: {str(e)}")
 
 
@app.post("/api/config")
async def update_config(
    num_questions: int   = 50,
    num_choices:   int   = 5,
    passing_score: float = 60.0,
    threshold:     float = 0.55,
):
    """Update konfigurasi processor (jumlah soal, pilihan, threshold)."""
    global processor
    processor = OMRProcessor(
        num_questions=num_questions,
        num_choices=num_choices,
        bubble_fill_threshold=threshold,
    )
    return {
        "success": True,
        "config": {
            "num_questions": num_questions,
            "num_choices":   num_choices,
            "passing_score": passing_score,
            "threshold":     threshold,
        }
    }
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
 