import test from 'node:test';
import assert from 'node:assert/strict';
import { calculateDemo } from '../docs/demo-calculator.mjs';

const compute = (subtotal, overrides = {}) => calculateDemo({ subtotal, member: true, couponClaimed: true, couponValid: true, ...overrides });
test('final price observes threshold boundaries and stacking order', () => {
  for (const [amount, expected] of [['199.99','179.99'],['200','162.00'],['299.99','251.99'],['300','207.00'],['400','297.00'],['600','477.00']]) {
    assert.equal(compute(amount).final, expected);
  }
});
test('only valid and claimed coupons are deducted', () => {
  assert.equal(compute('400', { couponClaimed: false }).final, '315.00');
  assert.equal(compute('400', { couponValid: false }).final, '315.00');
  assert.equal(compute('400', { member: false }).final, '330.00');
});
test('final half-up rounding is exact at cent boundaries', () => {
  assert.equal(compute('1.05').final, '0.95');
  assert.equal(compute('0').final, '0.00');
  assert.equal(compute('1000000').final, '899937.00');
});
test('invalid or out-of-range amounts are rejected', () => {
  for (const subtotal of ['', '-1', '1.005', 'NaN', 'Infinity', '1e3', '1000001']) {
    assert.throws(() => compute(subtotal));
  }
});
