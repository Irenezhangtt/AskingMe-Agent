"""Opt-in live vision smoke test using a generated, fictional offer (uses API credits).

Does not upload user documents. Requires ANTHROPIC_API_KEY and a vision-capable model.
"""
import asyncio
import io
import os
from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from rag.llm import PricingLLM
from rag.policy_uploads import OrderFacts, UploadedPriceRequest, calculate_uploaded_price, extract_policy, image_block

TEXT = '''FICTIONAL SEPTEMBER OFFER
Valid from 2026-09-01 through 2026-09-30 (inclusive).
Appliances only. Currency: CNY.
For an original subtotal of at least CNY 300, subtract CNY 50 once per order.
Then subtract one CNY 20 coupon if it is already claimed and valid.
Members then get 10% off the remaining amount. These three offers stack in that order.
Do not round intermediate amounts. Round the final price to two decimal places, half-up.
This final price is the merchandise subtotal only, excluding shipping and tax.'''


async def main():
    load_dotenv()
    kwargs = {'api_key': os.environ['ANTHROPIC_API_KEY'], 'timeout': 90.0, 'max_retries': 0}
    if os.getenv('ANTHROPIC_BASE_URL'):
        kwargs['base_url'] = os.environ['ANTHROPIC_BASE_URL']
    canvas = Image.new('RGB', (1600, 750), '#fffefa')
    font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    font = ImageFont.truetype(str(font_path), 28) if font_path.exists() else ImageFont.load_default(size=28)
    drawing = ImageDraw.Draw(canvas)
    lines = '\n'.join('\n'.join(textwrap.wrap(line, width=95)) for line in TEXT.splitlines())
    drawing.multiline_text((45, 45), lines, font=font, fill='#173f33', spacing=18)
    buffer = io.BytesIO()
    canvas.save(buffer, 'PNG')
    async with AsyncAnthropic(**kwargs) as client:
        llm = PricingLLM(client, os.getenv('ANSWER_MODEL', os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-6')))
        evidence = await extract_policy(llm, 'fictional-september-offer.png', image=image_block(buffer.getvalue()))
        assert evidence.date_scope == 'dated', evidence.model_dump(mode='json')
        assert str(evidence.effective_from) == '2026-09-01'
        assert str(evidence.effective_to) == '2026-09-30'
        assert 'half-up' in evidence.content.lower()
        # This is a known synthetic fixture, not automatic approval of user evidence.
        evidence.reviewed = True
        request = UploadedPriceRequest(policies=[evidence], order=OrderFacts(
            subtotal='400.00', currency='CNY', purchase_date='2026-09-22', category='Appliances',
            member=True, coupon_claimed=True, coupon_valid=True,
            details='The claimed, valid coupon amount is CNY 20. Calculate merchandise subtotal, excluding shipping and tax.'))
        result = await calculate_uploaded_price(llm, request)
        assert result.get('calculation', {}).get('final_price') == '297.00', result['answer']
        print('PASS: real screenshot -> vision transcription -> policy dates -> Decimal final price CNY 297.00')
        request.order.purchase_date = __import__('datetime').date(2026, 10, 1)
        expired = await calculate_uploaded_price(llm, request)
        assert expired['type'] == 'clarification' and len(expired['excluded']) == 1
        print('PASS: expired policy excluded before any calculation model call')


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as error:
        # Provider exception messages may contain endpoint details; don't print them.
        print(f'FAIL: {type(error).__name__}. Check the configured provider/model or fixture expectations.')
        raise SystemExit(1)
