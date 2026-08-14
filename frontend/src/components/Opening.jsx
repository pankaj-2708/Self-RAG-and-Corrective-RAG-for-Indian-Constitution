const STARTERS = [
  'What does Article 21 protect?',
  'Difference between fundamental rights and directive principles',
  'When can Article 356 be invoked?',
  'What does IPC Section 300 treat as murder?',
]

export default function Opening({ onPick }) {
  return (
    <div className="opening opening--minimal">
      <section className="opening__minimal">
        <p className="opening__minimal-kicker">Constitution of India assistant</p>
        <h1 className="opening__minimal-title">Samvidhan</h1>

        <div className="starters starters--minimal">
          {STARTERS.map((text) => (
            <button key={text} className="starter starter--minimal" onClick={() => onPick(text)}>
              {text}
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
