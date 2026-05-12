// src/components/LandingPage.tsx
import './LandingPage.css';
import { useState, type ChangeEvent } from 'react';

interface GradingResult {
  score: number;
  status: 'Lulus' | 'Tidak Lulus' | 'Error';
  correct_count: number;
  wrong_count: number;
  empty_count: number;
  total_questions: number;
  preview_image?: string;       // Base64 PNG dari backend
  details: {
    question: number;
    detected: string | null;
    correct: string | null;
    is_correct: boolean;
  }[];
  error?: string;
}

const API_URL = 'http://localhost:8000';

const LandingPage: React.FC = () => {
  const [file1, setFile1] = useState<File | null>(null); // Kunci jawaban
  const [file2, setFile2] = useState<File | null>(null); // Lembar siswa
  const [result, setResult] = useState<GradingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (
    e: ChangeEvent<HTMLInputElement>,
    setter: React.Dispatch<React.SetStateAction<File | null>>
  ) => {
    if (e.target.files && e.target.files.length > 0) {
      setter(e.target.files[0]);
    }
  };

  const handleProcess = async () => {
    if (!file1 || !file2) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Kirim kedua file ke backend via FormData
      const formData = new FormData();
      formData.append('answer_key_image', file1); // Nama harus match parameter di app.py
      formData.append('student_sheet', file2);

      const response = await fetch(`${API_URL}/api/grade`, {
        method: 'POST',
        body: formData,
        // Jangan set Content-Type manual — browser otomatis isi boundary untuk multipart
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || `Server error: ${response.status}`);
      }

      const data: GradingResult = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Gagal memproses lembar jawaban');
      }

      setResult(data);

    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Terjadi kesalahan tidak diketahui';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="landing-container">
      <nav className="navbar">
        <div className="logo">AutoGrade</div>
        <div className="nav-links">
          <a href="#">Fitur</a>
          <button className="btn-login">Masuk</button>
        </div>
      </nav>

      <header className="landing-hero">
        <h1>Koreksi Lembar Jawaban Lebih Cepat & Otomatis</h1>
        <p>Upload kunci jawaban dan lembar siswa, sistem akan mencocokkan dan memberi nilai otomatis.</p>
        <button className="btn-primary">Mulai Sekarang</button>
      </header>

      <section className="demo-section">
        <div className="card">
          <h2>Interactive Demo</h2>

          <div className="upload-container">
            <div className="upload-box">
              <p>Upload Kunci Jawaban</p>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleFileChange(e, setFile1)}
              />
              {file1 && <p className="file-name">📄 {file1.name}</p>}
            </div>
            <div className="upload-box">
              <p>Upload Lembar Siswa</p>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleFileChange(e, setFile2)}
              />
              {file2 && <p className="file-name">📄 {file2.name}</p>}
            </div>
          </div>

          <button
            className="btn-process"
            disabled={!file1 || !file2 || loading}
            onClick={handleProcess}
          >
            {loading ? 'Memproses...' : 'Proses Penilaian'}
          </button>

          {/* Error state */}
          {error && (
            <div className="error-card">
              <p>⚠️ {error}</p>
            </div>
          )}

          {/* Result */}
          {result && !error && (
            <div className="result-card">
              {/* Skor utama */}
              <div className="result-header">
                <p>Hasil: <strong>{result.score}/100</strong></p>
                <p className={result.status === 'Lulus' ? 'status-pass' : 'status-fail'}>
                  Status: {result.status}
                </p>
              </div>

              {/* Statistik ringkasan */}
              <div className="result-stats">
                <div className="stat-item">
                  <span className="stat-value correct">{result.correct_count}</span>
                  <span className="stat-label">Benar</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value wrong">{result.wrong_count}</span>
                  <span className="stat-label">Salah</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value empty">{result.empty_count}</span>
                  <span className="stat-label">Kosong</span>
                </div>
              </div>

              {/* Preview gambar hasil koreksi */}
              {result.preview_image && (
                <div className="preview-container">
                  <p className="preview-label">Preview Hasil Koreksi:</p>
                  <img
                    src={`data:image/png;base64,${result.preview_image}`}
                    alt="Hasil koreksi lembar jawaban"
                    className="preview-image"
                  />
                </div>
              )}

              {/* Tabel detail per soal */}
              {result.details.length > 0 && (
                <details className="detail-toggle">
                  <summary>Lihat Detail Per Soal ({result.total_questions} soal)</summary>
                  <table className="detail-table">
                    <thead>
                      <tr>
                        <th>No</th>
                        <th>Jawaban Siswa</th>
                        <th>Kunci</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.details.map((d) => (
                        <tr key={d.question} className={d.is_correct ? 'row-correct' : 'row-wrong'}>
                          <td>{d.question}</td>
                          <td>{d.detected ?? '–'}</td>
                          <td>{d.correct ?? '–'}</td>
                          <td>{d.is_correct ? '✓' : '✗'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
