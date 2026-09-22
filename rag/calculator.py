"""Bounded Decimal arithmetic for model-proposed final-price expressions.

Only decimal literals, +, -, *, /, parentheses, min and max are supported.
No Python eval, names, attributes, indexing, powers, or arbitrary calls.
"""
import ast
from decimal import Decimal, DecimalException, ROUND_HALF_UP, localcontext
import re


class CalculationError(ValueError):
    pass


def calculate_final_price(expression: str) -> dict:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 500:
        raise CalculationError('Provide a nonempty arithmetic expression of at most 500 characters')
    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, ValueError, RecursionError) as ex:
        raise CalculationError('Invalid arithmetic expression') from ex
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise CalculationError('Expression is too complex')
    steps = []

    def bounded(value):
        if not value.is_finite() or abs(value) > Decimal('1000000000000'):
            raise CalculationError('Amounts must be finite and no greater than one trillion in magnitude')
        return value

    def visit(node, depth=0):
        if depth > 16:
            raise CalculationError('Expression nesting is too deep')
        if isinstance(node, ast.Constant):
            literal = ast.get_source_segment(expression, node) or ''
            if not re.fullmatch(r'\d{1,13}(?:\.\d{1,12})?', literal):
                raise CalculationError('Use plain decimal amounts; exponent notation and strings are not supported')
            return bounded(Decimal(literal))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand, depth + 1)
            return -result if isinstance(node.op, ast.USub) else result
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            else:
                if right == 0:
                    raise CalculationError('Division by zero is not allowed')
                result = left / right
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'min', 'max'}:
            if node.keywords or not 2 <= len(node.args) <= 10:
                raise CalculationError('min/max require two to ten positional amounts')
            values = [visit(arg, depth + 1) for arg in node.args]
            result = min(values) if node.func.id == 'min' else max(values)
        else:
            raise CalculationError('Unsupported arithmetic operation')
        result = bounded(result)
        steps.append({'expression': ast.get_source_segment(expression, node), 'result': format(result, 'f')})
        return result

    try:
        with localcontext() as context:
            context.prec = 50
            raw = visit(tree.body)
            if raw < 0:
                raise CalculationError('A final price cannot be negative; confirm the applicable floor rule')
            price = raw.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    except DecimalException as ex:
        raise CalculationError('The arithmetic could not be evaluated safely') from ex
    return {'expression': expression, 'unrounded': format(raw, 'f'), 'final_price': format(price, '.2f'),
            'rounding': 'ROUND_HALF_UP', 'decimal_places': 2, 'steps': steps}
