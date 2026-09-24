const base = 'https://raw.githubusercontent.com/Irenezhangtt/AskingMe-Agent/main/examples/evaluation/golden/'
const family = document.querySelector('#family')
const outcome = document.querySelector('#outcome')
const search = document.querySelector('#search')
const list = document.querySelector('#case-list')
const detail = document.querySelector('#case-detail')
const count = document.querySelector('#count')
let cases = []
let selected = ''
const human = (text) => text.replaceAll('_', ' ')
const node = (tag, text, className) => {
  const element = document.createElement(tag)
  element.textContent = text
  if (className) element.className = className
  return element
}
function selectCase(item) {
  selected = item.id
  history.replaceState(null, '', `#${item.id}`)
  for (const button of list.querySelectorAll('button')) button.setAttribute('aria-current', String(button.dataset.id === selected))
  detail.replaceChildren(node('p', `${item.id} / ${human(item.family).toUpperCase()}`, 'eyebrow'), node('h2', human(item.family)))
  detail.append(node('p', item.question))
  const facts = node('div', '', 'gold-facts')
  for (const [label, value] of [['Original subtotal', `${item.order.currency} ${item.order.subtotal}`], ['Purchase date', item.order.purchase_date], ['Category', item.order.category], ['Member', item.order.member === null ? 'Unknown' : item.order.member ? 'Yes' : 'No'], ['Coupon claimed', item.order.coupon_claimed ? 'Yes' : 'No'], ['Labels', item.order.tags.join(', ')]]) {
    const fact = node('div', '')
    fact.append(node('small', label), node('span', value))
    facts.append(fact)
  }
  detail.append(facts)
  const expected = node('div', '', 'gold-outcome')
  expected.append(node('small', 'EXPECTED OUTCOME / SYNTHETIC LABEL'))
  if (item.expected.type === 'calculation') {
    expected.append(node('strong', `${item.expected.currency} ${item.expected.final_price}`), node('code', item.expected.expression), node('p', 'Independent exact-rational label, final half-up to two decimal places.'))
  } else {
    expected.append(node('strong', 'Clarification required'), node('p', item.expected.clarification_reason))
  }
  detail.append(expected, node('h3', 'Reference assertions'))
  const claims = node('ul', '')
  item.reference_claims.forEach((claim) => claims.append(node('li', claim)))
  detail.append(claims, node('h3', 'Policy evidence supplied to the agent'))
  for (const policy of item.policies) {
    const card = node('details', '')
    card.append(node('summary', policy.source_name), node('p', `Validity: ${policy.date_scope} · ${policy.effective_from || '?'} → ${policy.effective_to || '?'}`, 'gold-meta'), node('pre', policy.content, 'gold-policy'))
    detail.append(card)
  }
  detail.append(node('p', `Split: ${item.split} · Related sources: ${item.relevant_documents.join(', ')} · Human review pending.`, 'gold-meta'))
  if (!item.retrieval_evaluable) detail.append(node('p', 'This deliberately malformed upload is tested at the integrity gate and excluded from valid-document retrieval metrics.', 'gold-message'))
}
function render() {
  const query = search.value.trim().toLowerCase()
  const visible = cases.filter((item) => (!family.value || item.family === family.value) && (!outcome.value || item.expected.type === outcome.value) && (!query || `${item.id} ${item.question} ${item.policies.map((policy) => policy.content).join(' ')}`.toLowerCase().includes(query)))
  count.textContent = `${visible.length} of ${cases.length} cases`
  list.replaceChildren()
  for (const item of visible) {
    const button = node('button', '')
    button.type = 'button'
    button.dataset.id = item.id
    button.append(node('small', `${item.id} · ${human(item.family)}`), node('span', item.expected.type === 'calculation' ? `${item.order.subtotal} → ${item.expected.final_price} ${item.expected.currency}` : 'Ask / decline to calculate'))
    button.addEventListener('click', () => selectCase(item))
    list.append(button)
  }
  const next = visible.find((item) => item.id === selected) || visible[0]
  if (next) selectCase(next)
  else detail.replaceChildren(node('h2', 'No matching cases'), node('p', 'Choose another family or search term.'))
}
for (const control of [family, outcome, search]) control.addEventListener('input', render)
try {
  const response = await fetch(`${base}cases.jsonl`)
  if (!response.ok) throw new Error('Dataset unavailable')
  cases = (await response.text()).trim().split('\n').map((line) => JSON.parse(line))
  if (cases.length !== 500 || new Set(cases.map((item) => item.id)).size !== 500) throw new Error('Unexpected dataset version')
  for (const name of [...new Set(cases.map((item) => item.family))].sort()) {
    const option = node('option', human(name))
    option.value = name
    family.append(option)
  }
  selected = location.hash.slice(1)
  render()
} catch {
  count.textContent = 'Could not load the dataset.'
  detail.replaceChildren(node('h2', 'The dataset is temporarily unavailable'), node('p', 'Open the Dataset link above to inspect the published JSONL, or reload this page.'))
}
