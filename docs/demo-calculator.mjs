// Fixed, fictional Demo Promotion. All amounts use integer cents.
export function calculateDemo({ subtotal, member, couponClaimed, couponValid }) {
  if (!/^\d{1,7}(\.\d{1,2})?$/.test(String(subtotal))) {
    throw new Error('Enter an amount from 0 to 1,000,000 with at most two decimal places.');
  }
  const [whole, fraction = ''] = String(subtotal).split('.');
  const original = BigInt(whole) * 100n + BigInt(fraction.padEnd(2, '0'));
  if (original > 100000000n) throw new Error('The demo supports subtotals up to CNY 1,000,000.');
  const threshold = original >= 30000n ? 5000n : 0n;
  const coupon = original >= 20000n && couponClaimed && couponValid ? 2000n : 0n;
  const remainder = original - threshold - coupon;
  const base = remainder < 0n ? 0n : remainder;
  // A 90/100 multiplier, half-up once at the final cent; no floating-point arithmetic.
  const final = member ? (base * 90n + 50n) / 100n : base;
  const format = (cents) => `${cents / 100n}.${(cents % 100n).toString().padStart(2, '0')}`;
  return { original: format(original), threshold: format(threshold), coupon: format(coupon),
    afterDiscounts: format(base), memberSavings: format(base - final), final: format(final),
    savings: format(original - final), thresholdEligible: threshold > 0n,
    couponEligible: coupon > 0n, member, originalCents: Number(original), finalCents: Number(final) };
}
