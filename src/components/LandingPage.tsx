// src/components/LandingPage.tsx
import './LandingPage.css'; // Import CSS khusus komponen ini
import { useState, type ChangeEvent } from 'react';

interface GradingResult {
  score: number;
  status: 'Lulus' | 'Tidak Lulus';
}

const LandingPage: React.FC = () => {
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const [result, setResult] = useState<GradingResult | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>, setter: React.Dispatch<React.SetStateAction<File | null>>) => {
    if (e.target.files && e.target.files.length > 0) setter(e.target.files[0]);
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
              <input type="file" onChange={(e) => handleFileChange(e, setFile1)} />
            </div>
            <div className="upload-box">
              <p>Upload Lembar Siswa</p>
              <input type="file" onChange={(e) => handleFileChange(e, setFile2)} />
            </div>
          </div>
          
          <button 
            className="btn-process" 
            disabled={!file1 || !file2}
            onClick={() => setResult({ score: 85, status: 'Lulus' })}
          >
            Proses Penilaian
          </button>

          {result && (
            <div className="result-card">
              <p>Hasil: <strong>{result.score}/100</strong></p>
              <p className={result.status === 'Lulus' ? 'status-pass' : 'status-fail'}>
                Status: {result.status}
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
