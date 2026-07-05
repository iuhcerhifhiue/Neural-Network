"""Synthetic 5th-grade math corpus generator.

Produces lines the model learns to complete, e.g.:

    347 + 268 = 615
    900 - 512 = 388
    1/2 + 1/3 = 5/6
    3/4 - 1/4 = 1/2

Rules chosen to match 5th-grade level:
  * Addition / subtraction of whole numbers 0..999.
  * Subtraction never goes negative (first operand >= second).
  * Fractions use denominators 2..12, proper fractions as operands,
    answers reduced to lowest terms (and shown as a whole number when
    the denominator reduces to 1).
"""

import random
from fractions import Fraction

MAX_INT = 999
MAX_DEN = 12


def _fmt_frac(fr: Fraction) -> str:
    """Render a Fraction as '3/4', or '2' when it's a whole number."""
    if fr.denominator == 1:
        return str(fr.numerator)
    return f"{fr.numerator}/{fr.denominator}"


def _int_problem(rng: random.Random) -> str:
    if rng.random() < 0.5:
        a, b = rng.randint(0, MAX_INT), rng.randint(0, MAX_INT)
        return f"{a} + {b} = {a + b}"
    a = rng.randint(0, MAX_INT)
    b = rng.randint(0, a)                      # keep the result non-negative
    return f"{a} - {b} = {a - b}"


def _frac_problem(rng: random.Random) -> str:
    def rand_frac() -> Fraction:
        d = rng.randint(2, MAX_DEN)
        n = rng.randint(1, d - 1)              # proper fraction: 0 < n < d
        return Fraction(n, d)

    f1, f2 = rand_frac(), rand_frac()
    if rng.random() < 0.5:
        result = f1 + f2
        op = "+"
    else:
        if f2 > f1:                            # avoid negative answers
            f1, f2 = f2, f1
        result = f1 - f2
        op = "-"
    return f"{_fmt_frac(f1)} {op} {_fmt_frac(f2)} = {_fmt_frac(result)}"


def generate_lines(n: int, seed: int = 0, frac_ratio: float = 0.35) -> list[str]:
    """Return `n` unique math problem strings."""
    rng = random.Random(seed)
    seen: set[str] = set()
    lines: list[str] = []
    while len(lines) < n:
        line = _frac_problem(rng) if rng.random() < frac_ratio else _int_problem(rng)
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def generate_text(n: int, seed: int = 0, frac_ratio: float = 0.35) -> str:
    """Return a full corpus string (one problem per line, trailing newline)."""
    return "\n".join(generate_lines(n, seed, frac_ratio)) + "\n"


if __name__ == "__main__":
    # Quick peek at what the data looks like.
    for line in generate_lines(10, seed=42):
        print(line)
