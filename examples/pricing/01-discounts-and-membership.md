# Demo Promotion Final Price Rules

These fictional rules apply only to the Appliances category during the Demo Promotion. They do not represent any real shopping platform. All amounts are in CNY.

## Calculation order

Start with the original item subtotal A, apply the threshold discount, subtract an eligible category coupon, and then apply the member discount to the remaining amount. Shipping is excluded. The final price cannot be negative. Round only the final result to two decimal places using round-half-up; do not round intermediate results.

## Threshold discount

If the original item subtotal A is at least CNY 300, subtract CNY 50. Otherwise, no threshold discount applies. Apply this discount only once per order, not once per CNY 300 spent. Check eligibility against the original subtotal before any discounts.

D = (50 if A >= 300 else 0)

## Member discount

After subtracting the threshold discount and category coupon, members receive 10% off the remaining amount. Nonmembers receive no additional discount.

Price = round_half_up(max(0, (A - (50 if A >= 300 else 0) - C)) * (0.9 if member else 1), 2)

This is explanatory pseudocode. A is the original item subtotal, C is the eligible category coupon amount, and member indicates membership eligibility. round_half_up denotes decimal half-up rounding, not Python's built-in round function.

### Worked example

An appliance has an original price of CNY 400. The customer has a valid, claimed CNY 20 category coupon and is a member. First, 400 - 50 = 350. Then, 350 - 20 = 330. Finally, 330 x 0.9 = CNY 297. A nonmember pays CNY 330.
