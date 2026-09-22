import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const GITHUB_URL = 'https://github.com/Irenezhangtt/AskingMe-Agent'

const welcomeMessage = {
  id: 'welcome',
  role: 'assistant',
  content:
    'Hello, I am AskingMe, your Final Price AI Agent. I use approved rules to work out applicable discounts and a Decimal calculation tool to compute supported final prices, with calculation steps and source citations. Try asking: During the Demo Promotion, what is the final price of a CNY 400 appliance if I have a valid, claimed CNY 20 coupon and a membership?',
}

function loadMessages() {
  try {
    const stored = JSON.parse(
      localStorage.getItem('askingme_pricing_en_messages') || 'null',
    )
    return Array.isArray(stored) && stored.length ? stored : [welcomeMessage]
  } catch {
    return [welcomeMessage]
  }
}

function createId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }

  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

function createUserId() {
  const stored = localStorage.getItem('askingme_user_id')
  if (stored) return stored
  const next = `web-${createId()}`
  localStorage.setItem('askingme_user_id', next)
  return next
}

async function readApiError(response) {
  const payload = await response.json().catch(() => null)
  const detail = payload?.detail
  if (typeof detail === 'string') return detail
  if (detail?.message) return detail.message
  return payload?.message || `Request failed (${response.status})`
}

function RuleFilters({ value, onChange }) {
  return <div className="pricing-filters">
    <label>Product category<input value={value.category} placeholder="Optional, e.g. Appliances" onChange={(e) => onChange({ ...value, category: e.target.value })} /></label>
    <label>Promotion<input value={value.event} placeholder="Optional, e.g. Demo Promotion" onChange={(e) => onChange({ ...value, event: e.target.value })} /></label>
    <label>Applicable date<input type="date" value={value.as_of} onChange={(e) => onChange({ ...value, as_of: e.target.value })} /></label>
  </div>
}

function App() {
  const [activeView, setActiveView] = useState('chat')
  const [messages, setMessages] = useState(loadMessages)
  const [input, setInput] = useState('')
  const [convId, setConvId] = useState(
    () => localStorage.getItem('askingme_pricing_en_conv_id') || null,
  )
  const [userId] = useState(createUserId)
  const [isSending, setIsSending] = useState(false)
  const [streamPhase, setStreamPhase] = useState('')
  const [error, setError] = useState('')
  const [serviceStatus, setServiceStatus] = useState('checking')
  const [quota, setQuota] = useState(null)
  const [adminKey, setAdminKey] = useState(
    () => sessionStorage.getItem('askingme_admin_key') || '',
  )
  const [adminDraft, setAdminDraft] = useState('')
  const [adminError, setAdminError] = useState('')
  const [knowledgeStats, setKnowledgeStats] = useState(null)
  const [knowledgeDocuments, setKnowledgeDocuments] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [retrievalFilters, setRetrievalFilters] = useState({ category: '', event: '', as_of: '' })
  const [isSearching, setIsSearching] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [operations, setOperations] = useState(null)
  const [operationsError, setOperationsError] = useState('')
  const [isRefreshingOperations, setIsRefreshingOperations] = useState(false)
  const listRef = useRef(null)

  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [messages, isSending])

  useEffect(() => {
    localStorage.setItem('askingme_pricing_en_messages', JSON.stringify(messages))
  }, [messages])

  useEffect(() => {
    let active = true

    async function checkHealth() {
      try {
        const response = await fetch(`${API_BASE_URL}/health`)
        const data = await response.json()
        if (!active) return
        setServiceStatus(
          response.ok && data.status === 'ok' ? 'online' : 'degraded',
        )
      } catch {
        if (active) setServiceStatus('offline')
      }
    }

    async function checkQuota() {
      try {
        const response = await fetch(`${API_BASE_URL}/demo/quota`)
        if (!response.ok) return
        const data = await response.json()
        if (active) setQuota(data)
      } catch {
        // Quota status does not affect core chat functionality.
      }
    }

    checkHealth()
    checkQuota()
    const timer = window.setInterval(checkHealth, 30000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  async function sendMessage(event) {
    event.preventDefault()
    const message = input.trim()
    if (!message || isSending) return

    setMessages((current) => [
      ...current,
      { id: createId(), role: 'user', content: message },
    ])
    setInput('')
    setError('')
    setIsSending(true)
    setStreamPhase('Preparing your request')
    const assistantId = createId()
    setMessages((current) => [
      ...current,
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        streaming: true,
      },
    ])

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          user_id: userId,
          conv_id: convId,
          filters: Object.fromEntries(Object.entries(retrievalFilters).filter(([, value]) => value)),
        }),
      })
      const remaining = response.headers.get('X-RateLimit-Remaining')
      const limit = response.headers.get('X-RateLimit-Limit')
      if (remaining !== null && limit !== null) {
        setQuota((current) => ({
          ...current,
          enabled: true,
          limit: Number(limit),
          remaining: Number(remaining),
        }))
      }
      if (!response.ok) {
        if (response.status === 429) {
          const payload = await response.json().catch(() => null)
          throw new Error(
            payload?.detail ||
              "Today's free demo quota has been used. Please try again after it resets.",
          )
        }
        if (response.status >= 500) {
          throw new Error('The AI service is temporarily busy. Please try again shortly.')
        }
        throw new Error(`The request could not be completed (${response.status}).`)
      }

      if (!response.body) {
        throw new Error('The browser could not open the response stream.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let responseMeta = null

      while (true) {
        const { value, done } = await reader.read()
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim()) continue
          const eventData = JSON.parse(line)
          if (eventData.type === 'phase') {
            setStreamPhase(eventData.detail)
          } else if (eventData.type === 'meta') {
            responseMeta = eventData.data
            setConvId(responseMeta.conv_id)
            localStorage.setItem('askingme_pricing_en_conv_id', responseMeta.conv_id)
          } else if (eventData.type === 'answer') {
            setStreamPhase('Writing an answer grounded in the pricing rules')
            setMessages((current) =>
              current.map((item) =>
                item.id === assistantId
                  ? { ...item, content: item.content + eventData.delta }
                  : item,
              ),
            )
          } else if (eventData.type === 'error') {
            throw new Error(eventData.detail || 'The response stream failed.')
          }
        }
        if (done) break
      }

      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                streaming: false,
                meta: responseMeta
                  ? {
                      intent: responseMeta.intent,
                      agent: responseMeta.agent_type,
                      knowledgeUsed: responseMeta.knowledge_used,
                      escalated: responseMeta.escalated,
                      latency: responseMeta.latency_ms,
                      cacheHit: responseMeta.cache_hit,
                      sources: responseMeta.sources || [],
                    }
                  : undefined,
              }
            : item,
        ),
      )
    } catch (requestError) {
      setMessages((current) =>
        current.filter((item) => item.id !== assistantId),
      )
      setError(
        requestError instanceof TypeError
          ? 'The service is currently unreachable. Check your connection and try again.'
          : `Message failed: ${requestError.message}`,
      )
    } finally {
      setIsSending(false)
      setStreamPhase('')
    }
  }

  function resetConversation() {
    localStorage.removeItem('askingme_pricing_en_conv_id')
    localStorage.removeItem('askingme_pricing_en_messages')
    setConvId(null)
    setMessages([welcomeMessage])
    setInput('')
    setError('')
    setStreamPhase('')
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  function adminHeaders(extra = {}) {
    return { ...extra, 'X-Admin-Key': adminKey }
  }

  async function unlockAdmin(event) {
    event.preventDefault()
    const candidate = adminDraft.trim()
    if (!candidate) return
    setAdminError('')
    try {
      const [response, documentsResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/knowledge/stats`, {
          headers: { 'X-Admin-Key': candidate },
        }),
        fetch(`${API_BASE_URL}/knowledge/documents`, {
          headers: { 'X-Admin-Key': candidate },
        }),
      ])
      if (!response.ok) throw new Error(await readApiError(response))
      if (!documentsResponse.ok) throw new Error(await readApiError(documentsResponse))
      const data = await response.json()
      const registry = await documentsResponse.json()
      sessionStorage.setItem('askingme_admin_key', candidate)
      setAdminKey(candidate)
      setAdminDraft('')
      setKnowledgeStats(data)
      setKnowledgeDocuments(registry.documents || [])
    } catch (requestError) {
      setAdminError(
        requestError instanceof TypeError
          ? 'The backend is unreachable.'
          : requestError.message,
      )
    }
  }

  function lockAdmin() {
    sessionStorage.removeItem('askingme_admin_key')
    setAdminKey('')
    setAdminDraft('')
    setKnowledgeStats(null)
    setKnowledgeDocuments([])
    setSearchResults([])
    setOperations(null)
  }

  async function refreshKnowledgeStats() {
    if (!adminKey) return
    const [statsResponse, documentsResponse] = await Promise.all([
      fetch(`${API_BASE_URL}/knowledge/stats`, { headers: adminHeaders() }),
      fetch(`${API_BASE_URL}/knowledge/documents`, { headers: adminHeaders() }),
    ])
    if (statsResponse.status === 401 || documentsResponse.status === 401) {
      lockAdmin()
      throw new Error('Your admin session expired. Enter the admin key again.')
    }
    if (!statsResponse.ok) throw new Error(await readApiError(statsResponse))
    if (!documentsResponse.ok) throw new Error(await readApiError(documentsResponse))
    const stats = await statsResponse.json()
    const registry = await documentsResponse.json()
    setKnowledgeStats(stats)
    setKnowledgeDocuments(registry.documents || [])
    return stats
  }

  async function searchKnowledge(event) {
    event.preventDefault()
    const query = searchQuery.trim()
    if (!query || isSearching) return
    setIsSearching(true)
    setAdminError('')
    try {
      const params = new URLSearchParams({ query, top_k: '5', ...Object.fromEntries(Object.entries(retrievalFilters).filter(([, value]) => value)) })
      const response = await fetch(`${API_BASE_URL}/search?${params}`, {
        method: 'POST',
        headers: adminHeaders(),
      })
      if (!response.ok) throw new Error(await readApiError(response))
      const data = await response.json()
      setSearchResults(Array.isArray(data.results) ? data.results : [])
    } catch (requestError) {
      setAdminError(requestError.message)
    } finally {
      setIsSearching(false)
    }
  }

  async function uploadKnowledge(event) {
    event.preventDefault()
    const file = event.currentTarget.elements.policyFile.files?.[0]
    if (!file || isUploading) return
    setIsUploading(true)
    setUploadStatus('')
    setAdminError('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      for (const field of ['category', 'event', 'effective_from', 'effective_to']) {
        const value = event.currentTarget.elements[field].value
        if (value) formData.append(field, value)
      }
      formData.append('title', event.currentTarget.elements.policyTitle.value)
      formData.append('version', event.currentTarget.elements.policyVersion.value || '1.0')
      formData.append('approve', event.currentTarget.elements.approveNow.checked ? 'true' : 'false')
      formData.append('replaces_document_id', event.currentTarget.elements.replacesDocument.value)
      const response = await fetch(`${API_BASE_URL}/knowledge/upload`, {
        method: 'POST',
        headers: adminHeaders(),
        body: formData,
      })
      if (!response.ok) throw new Error(await readApiError(response))
      const data = await response.json()
      setUploadStatus(
        `${data.message}. ${data.added_chunks} chunks added as ${
          data.documents?.[0]?.status || 'draft'
        }.`,
      )
      event.currentTarget.reset()
      await refreshKnowledgeStats()
    } catch (requestError) {
      setAdminError(requestError.message)
    } finally {
      setIsUploading(false)
    }
  }

  async function manageDocument(action, documentId) {
    const label = action === 'approve' ? 'approve' : 'permanently delete'
    if (!window.confirm(`Are you sure you want to ${label} this document version?`)) return
    setAdminError('')
    setUploadStatus('')
    try {
      const response = await fetch(
        `${API_BASE_URL}/knowledge/documents/${encodeURIComponent(documentId)}${
          action === 'approve' ? '/approve' : ''
        }`,
        {
          method: action === 'approve' ? 'POST' : 'DELETE',
          headers: adminHeaders(),
        },
      )
      if (!response.ok) throw new Error(await readApiError(response))
      const data = await response.json()
      setUploadStatus(data.message)
      await refreshKnowledgeStats()
    } catch (requestError) {
      setAdminError(requestError.message)
    }
  }

  async function refreshOperations() {
    if (!adminKey || isRefreshingOperations) return
    setIsRefreshingOperations(true)
    setOperationsError('')
    try {
      const [healthResponse, monitorResponse, skillsResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/health`),
        fetch(`${API_BASE_URL}/monitor`, { headers: adminHeaders() }),
        fetch(`${API_BASE_URL}/skills`, { headers: adminHeaders() }),
      ])
      if (!healthResponse.ok) throw new Error(await readApiError(healthResponse))
      if (!monitorResponse.ok) throw new Error(await readApiError(monitorResponse))
      if (!skillsResponse.ok) throw new Error(await readApiError(skillsResponse))
      setOperations({
        health: await healthResponse.json(),
        monitor: await monitorResponse.json(),
        skills: await skillsResponse.json(),
        refreshedAt: new Date().toLocaleTimeString(),
      })
    } catch (requestError) {
      setOperationsError(requestError.message)
    } finally {
      setIsRefreshingOperations(false)
    }
  }

  function openView(view) {
    setActiveView(view)
    setAdminError('')
    if (view === 'knowledge' && adminKey && !knowledgeStats) {
      refreshKnowledgeStats().catch((requestError) =>
        setAdminError(requestError.message),
      )
    }
    if (view === 'operations' && adminKey && !operations) {
      refreshOperations()
    }
  }

  const quotaExhausted =
    quota?.enabled &&
    (quota.available === false || Number(quota.remaining) <= 0)

  const navigation = [
    { id: 'chat', icon: '✦', label: 'Final Price Agent', detail: 'Calculate & explain' },
    { id: 'knowledge', icon: '⌕', label: 'Knowledge lab', detail: 'Search & import' },
    { id: 'operations', icon: '◫', label: 'Operations', detail: 'Health & agents' },
  ]

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">A</div>
          <div>
            <strong>AskingMe Agent</strong>
            <span>Final Price AI Agent</span>
          </div>
        </div>

        <nav className="workspace-nav" aria-label="Workspace">
          {navigation.map((item) => (
            <button
              className={activeView === item.id ? 'active' : ''}
              type="button"
              onClick={() => openView(item.id)}
              key={item.id}
            >
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>
              <span>
                <strong>{item.label}</strong>
                <small>{item.detail}</small>
              </span>
            </button>
          ))}
        </nav>

        {activeView === 'chat' && (
          <button className="new-chat" type="button" onClick={resetConversation}>
            <span aria-hidden="true">＋</span>
            New conversation
          </button>
        )}

        {activeView === 'chat' && <div className="sidebar-card">
          <span className="eyebrow">Current conversation</span>
          <strong>{convId ? 'Memory connected' : 'Ready to start'}</strong>
          <p>
            {convId
              ? `Conversation ${convId.slice(0, 8)}…`
              : 'A conversation ID is created after your first message'}
          </p>
        </div>}

        {quota?.enabled && (
          <div className="sidebar-card quota-card">
            <span className="eyebrow">Free demo quota</span>
            <strong>
              {quota.available === false ? 0 : quota.remaining} requests left today
            </strong>
            <p>Up to {quota.limit} per visitor each day · resets at 00:00 UTC</p>
          </div>
        )}

        <div className="demo-card">
          <strong>About this demo</strong>
          <p>Calculate final prices from discount rules, with Decimal arithmetic and cited evidence.</p>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer noopener">
            View the GitHub project ↗
          </a>
        </div>

        <div className="sidebar-footer">
          <span className="status-dot" />
          Free public demo
        </div>
      </aside>

      {activeView === 'chat' && <section className="chat-panel">
        <header className="chat-header">
          <div>
            <span className="eyebrow">Final Price AI Agent</span>
            <h1>What is your final price?</h1>
          </div>
          <div className={`online-badge ${serviceStatus}`}>
            <span className="status-dot" />
            {
              {
                checking: 'Checking service',
                online: 'Service online',
                degraded: 'Service degraded',
                offline: 'Backend offline',
              }[serviceStatus]
            }
          </div>
        </header>

        <div className="message-list" ref={listRef} aria-live="polite">
          {messages.map((message) => (
            <article
              className={`message-row ${message.role}`}
              key={message.id}
            >
              <div className="avatar" aria-hidden="true">
                {message.role === 'assistant' ? 'A' : 'You'}
              </div>
              <div className="message-content">
                <div className="message-author">
                  {message.role === 'assistant' ? 'AskingMe Agent' : 'You'}
                </div>
                <div className={`bubble markdown-body ${message.streaming ? 'streaming' : ''}`}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      a: ({ children, ...props }) => (
                        <a
                          {...props}
                          target="_blank"
                          rel="noreferrer noopener"
                        >
                          {children}
                        </a>
                      ),
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                  {message.streaming && !message.content && (
                    <div className="stream-status">
                      <span className="stream-spinner" />
                      {streamPhase || 'Reviewing the relevant policy'}
                    </div>
                  )}
                </div>
                {message.meta?.sources?.length > 0 && (
                  <details className="source-evidence"><summary>Rule sources ({message.meta.sources.length})</summary>
                    <ol>{message.meta.sources.map((source, index) => <li key={source.chunk_id || index}>
                      {source.title} · {source.heading_path} · v{source.version}<br /><small>{source.source_name}</small>
                    </li>)}</ol>
                  </details>
                )}
                {message.meta &&
                  (message.meta.knowledgeUsed || message.meta.escalated) && (
                  <div className="message-meta">
                    {message.meta.knowledgeUsed && <span>Knowledge base</span>}
                    {message.meta.escalated && (
                      <span className="warning">Human review recommended</span>
                    )}
                  </div>
                )}
              </div>
            </article>
          ))}

        </div>

        <div className="composer-wrap">
          <RuleFilters value={retrievalFilters} onChange={setRetrievalFilters} />
          {quotaExhausted && (
            <div className="quota-message">
              Today&apos;s free demo quota is exhausted and resets at 00:00 UTC. You can still review this conversation and the source code.
            </div>
          )}
          {error && <div className="error-message">{error}</div>}
          <form className="composer" onSubmit={sendMessage}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                quotaExhausted
                  ? "Today's demo quota is exhausted"
                  : 'Example: A CNY 400 appliance, CNY 50 off orders of CNY 300 or more, a valid CNY 20 coupon, and 10% off for members. What is the final price?'
              }
              rows="1"
              disabled={isSending || quotaExhausted}
              aria-label="Chat message"
            />
            <button
              className="send-button"
              type="submit"
              disabled={!input.trim() || isSending || quotaExhausted}
              aria-label="Send message"
            >
              <span aria-hidden="true">↑</span>
            </button>
          </form>
          <p className="composer-hint">
            Enter to send · Shift + Enter for a new line · AI responses may contain errors
          </p>
        </div>
      </section>}

      {activeView === 'knowledge' && (
        <section className="workspace-panel">
          <WorkspaceHeader
            eyebrow="Knowledge management"
            title="Pricing knowledge lab"
            description="Search pricing rules, inspect ranked evidence, and import or approve rule documents."
            serviceStatus={serviceStatus}
          />
          {!adminKey ? (
            <AdminGate
              draft={adminDraft}
              setDraft={setAdminDraft}
              error={adminError}
              onSubmit={unlockAdmin}
            />
          ) : (
            <div className="workspace-scroll">
              <div className="workspace-toolbar">
                <div>
                  <span className="eyebrow">Admin session</span>
                  <strong>Internal tools unlocked</strong>
                </div>
                <button className="text-button" type="button" onClick={lockAdmin}>
                  Lock workspace
                </button>
              </div>

              {adminError && <div className="error-message wide">{adminError}</div>}
              {uploadStatus && <div className="success-message">{uploadStatus}</div>}

              <div className="stat-grid">
                <StatCard
                  label="Indexed chunks"
                  value={knowledgeStats?.total_chunks ?? '—'}
                  detail="Available to semantic retrieval"
                />
                <StatCard
                  label="Approved policies"
                  value={knowledgeStats?.status_counts?.approved ?? '—'}
                  detail={`${knowledgeStats?.status_counts?.draft ?? 0} drafts awaiting review`}
                />
                <StatCard label="Accepted files" value="PDF · DOCX · TXT" detail="Also MD and JSON · maximum 10 MB" />
              </div>

              <div className="console-grid">
                <section className="console-card">
                  <span className="eyebrow">Retrieval playground</span>
                  <h2>Search pricing rules</h2>
                  <RuleFilters value={retrievalFilters} onChange={setRetrievalFilters} />
                  <p>See the chunks that query rewriting and reranking select before they reach the answering agent.</p>
                  <form className="console-form" onSubmit={searchKnowledge}>
                    <label htmlFor="knowledge-query">Pricing question or search phrase</label>
                    <div className="inline-control">
                      <input
                        id="knowledge-query"
                        value={searchQuery}
                        onChange={(event) => setSearchQuery(event.target.value)}
                        placeholder="Example: Can the CNY 50 threshold discount be combined with a CNY 20 category coupon?"
                      />
                      <button type="submit" disabled={!searchQuery.trim() || isSearching}>
                        {isSearching ? 'Searching…' : 'Run search'}
                      </button>
                    </div>
                  </form>
                </section>

                <section className="console-card">
                  <span className="eyebrow">Document ingestion</span>
                  <h2>Import pricing rules</h2>
                  <p>Upload pricing rules. Markdown headings and complete formulas are preserved during chunking.</p>
                  <form className="console-form" onSubmit={uploadKnowledge}>
                    <div className="form-grid">
                      <label>
                        Rule title
                        <input name="policyTitle" placeholder="Uses filename when empty" />
                      </label>
                      <label>
                        Version
                        <input name="policyVersion" defaultValue="1.0" placeholder="Example: 2026.2" />
                      </label>
                    </div>
                    <div className="form-grid">
                      <label>Product category<input name="category" placeholder="e.g. Appliances" /></label>
                      <label>Promotion<input name="event" placeholder="e.g. Demo Promotion" /></label>
                      <label>Effective from<input name="effective_from" type="date" /></label>
                      <label>Effective until<input name="effective_to" type="date" /></label>
                    </div>
                    <label>
                      Replaces an earlier version
                      <select name="replacesDocument" defaultValue="">
                        <option value="">No replacement</option>
                        {knowledgeDocuments.map((document) => (
                          <option value={document.document_id} key={document.document_id}>
                            {document.title} · {document.version}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="file-drop" htmlFor="policy-file">
                      <span>Choose a policy file</span>
                      <small>PDF, DOCX, TXT, MD, or JSON · up to 10 MB</small>
                      <input id="policy-file" name="policyFile" type="file" accept=".pdf,.docx,.txt,.md,.json" required />
                    </label>
                    <label className="checkbox-control">
                      <input type="checkbox" name="approveNow" />
                      <span>Approve immediately (otherwise import as draft)</span>
                    </label>
                    <button type="submit" disabled={isUploading}>
                      {isUploading ? 'Importing…' : 'Import into knowledge base'}
                    </button>
                  </form>
                </section>
              </div>

              <section className="results-card">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">Ranked evidence</span>
                    <h2>{searchResults.length ? `${searchResults.length} results` : 'No search run yet'}</h2>
                  </div>
                </div>
                {searchResults.length ? (
                  <div className="result-list">
                    {searchResults.map((result, index) => (
                      <article className="result-item" key={`${result.title}-${index}`}>
                        <div className="result-rank">{String(index + 1).padStart(2, '0')}</div>
                        <div>
                          <div className="result-title">
                            <strong>{result.title || 'Untitled policy'}</strong>
                            {result.score !== undefined && <span>Score {Number(result.score).toFixed(3)}</span>}
                          </div>
                          <small>{result.heading_path} · {result.source_name} · v{result.version}</small>
                          <p>{result.content || 'No content returned.'}</p>
                        </div>
                      </article>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state">Run a search to inspect matching rules and their sources.</div>
                )}
              </section>

              <section className="results-card registry-card">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">Document lifecycle</span>
                    <h2>{knowledgeDocuments.length} document versions</h2>
                  </div>
                </div>
                {knowledgeDocuments.length ? (
                  <div className="document-table-wrap">
                    <table className="document-table">
                      <thead>
                        <tr>
                          <th>Policy</th>
                          <th>Version</th>
                          <th>Status</th>
                          <th>Chunks</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {knowledgeDocuments.map((document) => (
                          <tr key={document.document_id}>
                            <td>
                              <strong>{document.title}</strong>
                              <small>{document.source_name}</small>
                            </td>
                            <td>{document.version}</td>
                            <td><span className={`status-pill ${document.status}`}>{document.status}</span></td>
                            <td>{document.total_chunks}</td>
                            <td>
                              <div className="table-actions">
                                {document.status === 'draft' && (
                                  <button type="button" onClick={() => manageDocument('approve', document.document_id)}>
                                    Approve
                                  </button>
                                )}
                                <button className="danger-button" type="button" onClick={() => manageDocument('delete', document.document_id)}>
                                  Delete
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="empty-state">No document versions are registered.</div>
                )}
              </section>
            </div>
          )}
        </section>
      )}

      {activeView === 'operations' && (
        <section className="workspace-panel">
          <WorkspaceHeader
            eyebrow="System observability"
            title="Agent operations"
            description="Inspect service health, loaded specialists, and runtime metrics without exposing administrative APIs."
            serviceStatus={serviceStatus}
          />
          {!adminKey ? (
            <AdminGate
              draft={adminDraft}
              setDraft={setAdminDraft}
              error={adminError}
              onSubmit={unlockAdmin}
            />
          ) : (
            <div className="workspace-scroll">
              <div className="workspace-toolbar">
                <div>
                  <span className="eyebrow">Live status</span>
                  <strong>{operations?.refreshedAt ? `Updated at ${operations.refreshedAt}` : 'Ready to refresh'}</strong>
                </div>
                <div className="toolbar-actions">
                  <button className="text-button" type="button" onClick={lockAdmin}>Lock workspace</button>
                  <button type="button" onClick={refreshOperations} disabled={isRefreshingOperations}>
                    {isRefreshingOperations ? 'Refreshing…' : 'Refresh data'}
                  </button>
                </div>
              </div>
              {operationsError && <div className="error-message wide">{operationsError}</div>}
              <div className="stat-grid">
                <StatCard label="API status" value={operations?.health?.status || serviceStatus} detail="FastAPI application" />
                <StatCard label="Agent specialists" value={countObjectEntries(operations?.health?.agents)} detail="Available routing targets" />
                <StatCard label="Loaded skills" value={countObjectEntries(operations?.skills)} detail="Runtime skill registry" />
              </div>
              <div className="diagnostic-grid">
                <DiagnosticCard title="Service health" data={operations?.health} />
                <DiagnosticCard title="Agent monitor" data={operations?.monitor} />
                <DiagnosticCard title="Loaded skills" data={operations?.skills} />
              </div>
            </div>
          )}
        </section>
      )}
    </main>
  )
}

function WorkspaceHeader({ eyebrow, title, description, serviceStatus }) {
  return (
    <header className="workspace-header">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <div className={`online-badge ${serviceStatus}`}>
        <span className="status-dot" />
        {serviceStatus === 'online' ? 'Service online' : serviceStatus}
      </div>
    </header>
  )
}

function AdminGate({ draft, setDraft, error, onSubmit }) {
  return (
    <div className="admin-gate">
      <div className="gate-icon" aria-hidden="true">⌁</div>
      <span className="eyebrow">Restricted workspace</span>
      <h2>Enter the admin key</h2>
      <p>Knowledge ingestion and operational data are limited to authorized administrators. The key stays in this browser tab only.</p>
      <form onSubmit={onSubmit}>
        <input
          type="password"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="ADMIN_API_KEY"
          autoComplete="off"
        />
        <button type="submit" disabled={!draft.trim()}>Unlock tools</button>
      </form>
      {error && <div className="error-message wide">{error}</div>}
    </div>
  )
}

function StatCard({ label, value, detail }) {
  return (
    <article className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

function DiagnosticCard({ title, data }) {
  return (
    <section className="diagnostic-card">
      <h2>{title}</h2>
      {data ? <pre>{JSON.stringify(data, null, 2)}</pre> : <div className="empty-state">Refresh to load data.</div>}
    </section>
  )
}

function countObjectEntries(value) {
  if (Array.isArray(value)) return value.length
  if (value && typeof value === 'object') return Object.keys(value).length
  return value ?? '—'
}

export default App
