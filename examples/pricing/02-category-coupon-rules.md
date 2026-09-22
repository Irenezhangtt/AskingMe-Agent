# Demo Promotion Category Coupon Rules

These fictional rules apply only to the Appliances category during the Demo Promotion. All amounts are in CNY.

## CNY 20 appliance coupon

The CNY 20 appliance coupon requires an item in the Appliances category and an original subtotal of at least CNY 200 before discounts. The coupon must be valid and already claimed by the customer. All conditions must hold. Do not assume that a customer has claimed a coupon.

C = (20 if (category == "Appliances" and A >= 200 and coupon_valid and coupon_claimed) else 0)

## Stacking and exclusions

The CNY 20 appliance coupon can be combined with the CNY 50 discount on orders of CNY 300 or more and the 10% member discount. Only one category coupon can be used per order. Multiple category coupons cannot be combined. Coupons for other categories cannot be used on appliance orders.
