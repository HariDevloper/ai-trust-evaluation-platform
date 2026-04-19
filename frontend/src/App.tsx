import axios from 'axios'
import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

type DatasetAnalysis = {
  task_type: string
  input_fields: string[]
  output_field?: string | null
  sample_count: number
  confidence: number
  sensitive_fields: string[]
  field_types: Record<string, string>
}

type ResultPayload = {
  performance: Record<string, number>
  trust: Record<string, number>
  trust_score: number
  latencies: number[]
  detailed_outputs: Array<{
    index: number
    input: Record<string, unknown>
    expected: unknown
    prediction: unknown
    latency: number
    error?: string | null
  }>
}

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
})

function parseHeaders(raw: string): Record<string, string> {
  if (!raw.trim()) return {}
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .reduce<Record<string, string>>((acc, line) => {
      const [key, ...rest] = line.split(':')
      if (key && rest.length) acc[key.trim()] = rest.join(':').trim()
      return acc
    }, {})
}

function App() {
  const [phase, setPhase] = useState<'setup' | 'config' | 'running' | 'results'>('setup')
  const [datasetFile, setDatasetFile] = useState<File | null>(null)
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [modelType, setModelType] = useState<'local' | 'api'>('local')
  const [apiEndpoint, setApiEndpoint] = useState('')
  const [apiHeaders, setApiHeaders] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [modelId, setModelId] = useState('')
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null)
  const [hasLabels, setHasLabels] = useState(true)
  const [sampleLimit, setSampleLimit] = useState<number | ''>('')
  const [jobId, setJobId] = useState('')
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [results, setResults] = useState<ResultPayload | null>(null)
  const [jobs, setJobs] = useState<Array<Record<string, unknown>>>([])

  const loadJobs = async () => {
    const response = await api.get('/api/jobs')
    setJobs(response.data.jobs || [])
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadJobs()
    }, 0)
    return () => window.clearTimeout(timer)
  }, [])

  useEffect(() => {
    if (phase !== 'running' || !jobId) return
    const timer = window.setInterval(async () => {
      try {
        const statusResponse = await api.get(`/api/status/${jobId}`)
        const nextStatus = statusResponse.data.status
        setStatus(nextStatus)
        if (nextStatus === 'completed' || nextStatus === 'failed') {
          window.clearInterval(timer)
          const resultResponse = await api.get(`/api/results/${jobId}`)
          setResults(resultResponse.data.results ?? null)
          setError(resultResponse.data.error || '')
          setPhase('results')
          await loadJobs()
        }
      } catch {
        setError('Failed to poll job status.')
      }
    }, 2000)
    return () => window.clearInterval(timer)
  }, [jobId, phase])

  const latencyData = useMemo(
    () => (results?.latencies || []).map((latency, index) => ({ sample: index + 1, latency })),
    [results],
  )

  const trustData = useMemo(
    () =>
      Object.entries(results?.trust || {}).map(([name, value]) => ({
        name,
        value: Number(value),
      })),
    [results],
  )

  const uploadSetup = async () => {
    setError('')
    try {
      if (!datasetFile) throw new Error('Dataset file is required.')
      const datasetForm = new FormData()
      datasetForm.append('file', datasetFile)
      const datasetResponse = await api.post('/api/upload-dataset', datasetForm)
      setDatasetId(datasetResponse.data.dataset_id)
      setAnalysis(datasetResponse.data.analysis)

      if (modelType === 'local') {
        if (!modelFile) throw new Error('Local model file is required.')
        const modelForm = new FormData()
        modelForm.append('file', modelFile)
        modelForm.append('model_type', 'local')
        const modelResponse = await api.post('/api/upload-model', modelForm)
        setModelId(modelResponse.data.model_id)
      } else {
        const modelResponse = await api.post('/api/upload-model', {
          name: 'api-model',
          type: 'api',
          config: {
            endpoint: apiEndpoint,
            headers: parseHeaders(apiHeaders),
          },
        })
        setModelId(modelResponse.data.model_id)
      }
      setHasLabels(Boolean(datasetResponse.data.analysis?.output_field))
      setPhase('config')
    } catch (uploadError: unknown) {
      setError((uploadError as { message?: string }).message || 'Upload failed.')
    }
  }

  const startEvaluation = async () => {
    setError('')
    try {
      const response = await api.post('/api/run-test', {
        dataset_id: datasetId,
        model_id: modelId,
        has_labels: hasLabels,
        sample_limit: sampleLimit === '' ? null : Number(sampleLimit),
      })
      setJobId(response.data.job_id)
      setStatus(response.data.status)
      setPhase('running')
    } catch (runError: unknown) {
      setError((runError as { message?: string }).message || 'Failed to start evaluation.')
    }
  }

  const exportJson = () => {
    if (!results) return
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `results-${jobId || 'job'}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportCsv = () => {
    if (!results) return
    const rows = ['index,input,expected,prediction,latency,error']
    for (const row of results.detailed_outputs) {
      rows.push(
        [
          row.index,
          JSON.stringify(row.input),
          JSON.stringify(row.expected),
          JSON.stringify(row.prediction),
          row.latency,
          row.error ?? '',
        ]
          .map((v) => `"${String(v).replaceAll('"', '""')}"`)
          .join(','),
      )
    }
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `results-${jobId || 'job'}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <main className="app">
      <header className="header">
        <h1>AI Trust & Performance Evaluation Platform</h1>
        <p>Phase: {phase.toUpperCase()}</p>
      </header>
      {error ? <p className="error">{error}</p> : null}

      {phase === 'setup' && (
        <section className="panel">
          <h2>1) Setup</h2>
          <div className="grid">
            <label className="field">
              Dataset (CSV/JSON)
              <input type="file" accept=".csv,.json" onChange={(e) => setDatasetFile(e.target.files?.[0] || null)} />
            </label>
            <label className="field">
              Model Type
              <select value={modelType} onChange={(e) => setModelType(e.target.value as 'local' | 'api')}>
                <option value="local">Local model</option>
                <option value="api">API model</option>
              </select>
            </label>
            {modelType === 'local' ? (
              <label className="field">
                Local model (Python/JSON)
                <input type="file" onChange={(e) => setModelFile(e.target.files?.[0] || null)} />
              </label>
            ) : (
              <>
                <label className="field">
                  API endpoint
                  <input value={apiEndpoint} onChange={(e) => setApiEndpoint(e.target.value)} placeholder="https://..." />
                </label>
                <label className="field">
                  Headers (key:value per line)
                  <textarea value={apiHeaders} onChange={(e) => setApiHeaders(e.target.value)} rows={4} />
                </label>
              </>
            )}
          </div>
          <button onClick={() => void uploadSetup()}>Upload & Analyze</button>
        </section>
      )}

      {phase === 'config' && analysis && (
        <section className="panel">
          <h2>2) Configuration</h2>
          <pre className="analysis">{JSON.stringify(analysis, null, 2)}</pre>
          <div className="grid compact">
            <label className="field inline">
              <input type="checkbox" checked={hasLabels} onChange={(e) => setHasLabels(e.target.checked)} />
              Dataset has labels
            </label>
            <label className="field">
              Sample limit (optional)
              <input
                type="number"
                min={1}
                value={sampleLimit}
                onChange={(e) => setSampleLimit(e.target.value ? Number(e.target.value) : '')}
              />
            </label>
          </div>
          <button onClick={() => void startEvaluation()}>Start Evaluation</button>
        </section>
      )}

      {phase === 'running' && (
        <section className="panel">
          <h2>3) Running</h2>
          <p>Job: {jobId}</p>
          <p>Status: {status}</p>
          <div className="loader" />
        </section>
      )}

      {phase === 'results' && results && (
        <>
          <section className="panel">
            <h2>4) Results Dashboard</h2>
            <div className="cards">
              <article className="card trust-card">
                <h3>Trust Score</h3>
                <p className="big">{results.trust_score.toFixed(3)}</p>
              </article>
              {Object.entries(results.performance).map(([key, value]) => (
                <article className="card" key={key}>
                  <h3>{key}</h3>
                  <p>{Number(value).toFixed(4)}</p>
                </article>
              ))}
            </div>
            <div className="actions">
              <button onClick={exportJson}>Export JSON</button>
              <button onClick={exportCsv}>Export CSV</button>
            </div>
          </section>

          <section className="panel charts">
            <article>
              <h3>Latency per sample</h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={latencyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sample" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="latency" stroke="#5b67f1" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </article>
            <article>
              <h3>Trust metrics</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={trustData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#1f9c75" />
                </BarChart>
              </ResponsiveContainer>
            </article>
          </section>

          <section className="panel">
            <h3>Per-sample outputs</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Input</th>
                    <th>Expected</th>
                    <th>Prediction</th>
                    <th>Latency</th>
                    <th>Error</th>
                  </tr>
                </thead>
                <tbody>
                  {results.detailed_outputs.map((row) => (
                    <tr key={row.index}>
                      <td>{row.index}</td>
                      <td>{JSON.stringify(row.input)}</td>
                      <td>{JSON.stringify(row.expected)}</td>
                      <td>{JSON.stringify(row.prediction)}</td>
                      <td>{row.latency.toFixed(4)}</td>
                      <td>{row.error || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      <section className="panel">
        <h3>Job History</h3>
        <ul className="history">
          {jobs.map((job) => (
            <li key={String(job.job_id)}>
              {String(job.job_id)} — {String(job.status)} — trust: {String(job.trust_score ?? 'n/a')}
            </li>
          ))}
        </ul>
      </section>
    </main>
  )
}

export default App
