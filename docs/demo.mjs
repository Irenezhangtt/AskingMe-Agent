import { calculateDemo } from './demo-calculator.mjs';
const $ = (id) => document.getElementById(id);
const sourceRoot = 'https://github.com/Irenezhangtt/AskingMe-Agent/blob/main/examples/pricing/';
for (const link of document.querySelectorAll('[data-rule]')) {
  link.href = sourceRoot + (link.dataset.rule === '1' ? '01-discounts-and-membership.md' : '02-category-coupon-rules.md');
  link.target = '_blank'; link.rel = 'noopener';
}
function update() {
  try {
    const result = calculateDemo({ subtotal: $('subtotal').value, member: $('member').checked,
      couponClaimed: $('claimed').checked, couponValid: $('valid').checked });
    $('error').textContent = '';
    document.querySelector('.result-panel').classList.remove('invalid');
    for (const [id, key] of Object.entries({ final: 'final', savings: 'savings', 'step-original': 'original' })) $(id).textContent = result[key];
    $('original-caption').textContent = `from CNY ${result.original}`;
    $('step-threshold').textContent = `−${result.threshold}`;
    $('step-coupon').textContent = `−${result.coupon}`;
    $('step-member').textContent = `−${result.memberSavings}`;
    $('threshold-reason').textContent = result.thresholdEligible ? 'At least CNY 300 · applied once per order' : 'Not eligible · original subtotal is below CNY 300';
    $('coupon-reason').textContent = result.couponEligible ? 'At least CNY 200 · claimed and valid' : 'Requires CNY 200 subtotal, a claimed coupon, and validity';
    $('member-reason').textContent = result.member ? `10% off the remaining CNY ${result.afterDiscounts}` : 'Not a member · no additional discount';
    $('formula').textContent = `(${result.original} − ${result.threshold} − ${result.coupon}) × ${result.member ? '0.90' : '1.00'} = ${result.final}`;
    $('paid-bar').style.width = `${result.originalCents ? result.finalCents / result.originalCents * 100 : 0}%`;
    for (const button of document.querySelectorAll('[data-amount]')) button.classList.toggle('selected', button.dataset.amount === result.original);
  } catch (error) {
    $('error').textContent = error.message;
    for (const id of ['final','savings','step-original','step-threshold','step-coupon','step-member']) $(id).textContent = '—';
    $('original-caption').textContent = 'Waiting for a valid subtotal';
    $('paid-bar').style.width = '0%';
    $('formula').textContent = 'Enter a valid subtotal to calculate the final price.';
    document.querySelector('.result-panel').classList.add('invalid');
  }
}
for (const id of ['subtotal', 'member', 'claimed', 'valid']) $(id).addEventListener('input', update);
for (const button of document.querySelectorAll('[data-amount]')) button.addEventListener('click', () => { $('subtotal').value = button.dataset.amount; update(); });
update();
