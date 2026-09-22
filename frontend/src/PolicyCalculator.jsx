import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './PolicyCalculator.css'

const labels = (text) => text.split(',').map((tag) => tag.trim()).filter(Boolean)
const yesNo = (value) => value === '' ? null : value === 'yes'

async function apiResult(response) {
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = payload.detail
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map((item) => `${item.loc.slice(1).join(' / ')}: ${item.msg}`).join('; ')
      : `Request failed (${response.status})`)
  }
  return payload
}

export default function PolicyCalculator({ apiBase, adminKey }) {
  const [files, setFiles] = useState([])
  const [policies, setPolicies] = useState([])
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [order, setOrder] = useState({ subtotal: '', currency: 'CNY', purchase_date: '', category: '', tags: '', member: '', coupon_claimed: '', coupon_valid: '', details: '' })

  function updateOrder(key, value) {
    setOrder((previous) => ({ ...previous, [key]: value }))
    setResult(null)
  }
  function updatePolicy(index, patch) {
    setPolicies((previous) => previous.map((policy, i) => i === index ? { ...policy, reviewed: false, ...patch } : policy))
    setResult(null)
  }
  async function extract(event) {
    event.preventDefault()
    setBusy('Reading your policies and screenshots…')
    setError('')
    setPolicies([])
    setResult(null)
    try {
      if (!files.length || files.length > 8) throw new Error('Choose one to eight files.')
      if (files.some((file) => file.size > 5 * 1024 * 1024) || files.reduce((sum, file) => sum + file.size, 0) > 20 * 1024 * 1024) throw new Error('Use files up to 5 MB each and 20 MB in total.')
      const body = new FormData()
      files.forEach((file) => body.append('files', file))
      const data = await apiResult(await fetch(`${apiBase}/pricing/extract`, { method: 'POST', headers: { 'X-Admin-Key': adminKey }, body }))
      setPolicies(data.policies)
    } catch (ex) { setError(ex.message) }
    finally { setBusy('') }
  }
  async function calculate(event) {
    event.preventDefault()
    setBusy('Checking policy eligibility and calculating…')
    setError('')
    setResult(null)
    try {
      const data = await apiResult(await fetch(`${apiBase}/pricing/calculate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Admin-Key': adminKey },
        body: JSON.stringify({ policies, order: { ...order, tags: labels(order.tags),
          member: yesNo(order.member), coupon_claimed: yesNo(order.coupon_claimed), coupon_valid: yesNo(order.coupon_valid) } }),
      }))
      setResult(data)
    } catch (ex) { setError(ex.message) }
    finally { setBusy('') }
  }
  const reviewed = policies.length > 0 && policies.every((policy) => policy.reviewed)

  return <div className="workspace-scroll policy-workspace">
    <div className="policy-intro"><span className="eyebrow">YOUR POLICIES → YOUR FINAL PRICE</span><h2>Start with the actual offer.</h2><p>Upload the terms, check what they say, and see what you pay on your purchase date.</p></div>
    <ol className="policy-progress" aria-label="Calculation workflow"><li className={policies.length ? 'complete' : 'current'}>01 <span>Upload</span></li><li className={policies.length ? 'current' : ''}>02 <span>Review evidence</span></li><li className={reviewed ? 'current' : ''}>03 <span>Calculate</span></li></ol>
    <form className="policy-card" onSubmit={extract}><fieldset disabled={!!busy}>
      <div className="policy-heading"><span className="eyebrow">01 / POLICY FILES</span><span className="policy-badge">Private workspace</span></div>
      <label className="policy-drop"><strong>Choose policy documents or screenshots</strong><span>PNG, JPEG, WebP, text PDF, DOCX, TXT, MD or JSON · Up to 8 files</span><input type="file" multiple required accept=".png,.jpg,.jpeg,.webp,.pdf,.docx,.txt,.md,.json" onChange={(event) => { setFiles(Array.from(event.target.files)); setPolicies([]); setResult(null); setError('') }} /></label>
      {files.length > 0 && <ul className="policy-file-list">{files.map((file, i) => <li key={`${file.name}-${i}`}>{file.name} <small>{Math.ceil(file.size / 1024)} KB</small></li>)}</ul>}
      <p className="policy-help">Files are sent to your configured AI provider for extraction. They are not added to the shared knowledge base. Scanned PDFs should be uploaded as page screenshots. 5 MB per file, 20 MB total.</p>
      <button className="policy-primary" disabled={!files.length || !!busy}>Read policies</button>
    </fieldset></form>

    {policies.length > 0 && <section className="policy-card"><fieldset disabled={!!busy}>
      <span className="eyebrow">02 / REVIEW THE EVIDENCE</span><h2>Check the dates. Keep the small print.</h2><p className="policy-help">Check the extracted text against your originals. Tags describe a policy; its written conditions determine eligibility. Upload time is never treated as the effective date.</p>
      {policies.map((policy, index) => <article className="policy-evidence" key={index}>
        <div className="policy-heading"><h3>{policy.title}</h3><span className="policy-badge">{policy.source_name}</span></div>
        {policy.warnings.length > 0 && <ul className="policy-warnings">{policy.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul>}
        <div className="policy-fields">
          <label>Validity<select value={policy.date_scope} onChange={(event) => updatePolicy(index, { date_scope: event.target.value, ...(event.target.value !== 'dated' ? { effective_from: null, effective_to: null } : {}) })}><option value="unknown">Needs confirmation</option><option value="dated">Effective date range</option><option value="unrestricted">No date restriction — confirmed</option></select></label>
          {policy.date_scope === 'dated' && <><label>Effective from<input type="date" value={policy.effective_from || ''} onChange={(event) => updatePolicy(index, { effective_from: event.target.value || null })} /></label><label>Effective through (inclusive)<input type="date" value={policy.effective_to || ''} onChange={(event) => updatePolicy(index, { effective_to: event.target.value || null })} /></label></>}
          <label>Policy labels<input value={policy.tags.join(',')} placeholder="Appliances, Summer sale" onChange={(event) => updatePolicy(index, { tags: event.target.value.split(',') })} onBlur={() => updatePolicy(index, { tags: policy.tags.map((tag) => tag.trim()).filter(Boolean) })} /></label>
        </div>
        {policy.date_evidence && <blockquote className="policy-quote">{policy.date_evidence}</blockquote>}
        <details className="policy-transcript"><summary>Read or correct the extracted policy text</summary><textarea aria-label={`Policy text ${index + 1}`} rows="9" value={policy.content} onChange={(event) => updatePolicy(index, { content: event.target.value })} /></details>
        <label className="policy-check"><input type="checkbox" checked={policy.reviewed} onChange={(event) => updatePolicy(index, { reviewed: event.target.checked })} /><span>I checked the text, dates and labels against the original policy.</span></label>
      </article>)}
    </fieldset></section>}

    <form className="policy-card" onSubmit={calculate}><fieldset disabled={!!busy}>
      <span className="eyebrow">03 / ORDER DETAILS</span><h2>What are you buying, and when?</h2>
      <div className="policy-fields">
        <label>Original subtotal<input required inputMode="decimal" placeholder="400.00" value={order.subtotal} pattern="[0-9]{1,9}(\.[0-9]{1,2})?" onChange={(event) => updateOrder('subtotal', event.target.value)} /></label>
        <label>Currency<input required maxLength="3" pattern="[A-Z]{3}" value={order.currency} onChange={(event) => updateOrder('currency', event.target.value.toUpperCase())} /></label>
        <label>Purchase date<input required type="date" value={order.purchase_date} onChange={(event) => updateOrder('purchase_date', event.target.value)} /></label>
        <label>Product category<input required value={order.category} placeholder="e.g. Appliances" onChange={(event) => updateOrder('category', event.target.value)} /></label>
        <label>Product / promotion labels<input value={order.tags} placeholder="e.g. Brand A, Summer sale" onChange={(event) => updateOrder('tags', event.target.value)} /></label>
        {[['member', 'Member?'], ['coupon_claimed', 'Coupon already claimed?'], ['coupon_valid', 'Coupon valid for this order?']].map(([key, label]) => <label key={key}>{label}<select value={order[key]} onChange={(event) => updateOrder(key, event.target.value)}><option value="">Not specified</option><option value="yes">Yes</option><option value="no">No</option></select></label>)}
      </div>
      <label className="policy-details">Additional order facts or clarification<textarea rows="3" value={order.details} placeholder="Product model, coupon amount, shipping / tax scope, payment time and time zone, or an answer to the agent's question…" onChange={(event) => updateOrder('details', event.target.value)} /></label>
      <button className="policy-primary" disabled={!reviewed || !!busy}>Calculate my final price <span aria-hidden="true">↗</span></button>
      {!reviewed && <p className="policy-help">Upload and review your policies to enable calculation.</p>}
    </fieldset></form>

    {busy && <p className="policy-status" role="status">{busy}</p>}
    {error && <p className="error-message wide" role="alert">{error}</p>}
    {result && <section className={`policy-card policy-result ${result.type}`} aria-live="polite">
      <span className="eyebrow">{result.type === 'calculation' ? 'YOUR CALCULATION' : 'ONE MORE DETAIL'}</span>
      <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result.answer}</ReactMarkdown></div>
      {result.excluded.length > 0 && <div className="policy-excluded"><h3>Outside the purchase date</h3><ul>{result.excluded.map((item, i) => <li key={i}><strong>{item.source_name}</strong> — {item.reason}</li>)}</ul></div>}
      {result.sources.length > 0 && <details><summary>Inspect policy sources ({result.sources.length})</summary>{result.sources.map((source) => <div key={source.source_id}><h3>[{source.source_id}] {source.source_name}</h3><p className="policy-source-text">{source.content}</p></div>)}</details>}
    </section>}
  </div>
}
