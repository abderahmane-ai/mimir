"""Check a claim against evidence passages.

Install `mimirai[local]`, then run `make example NAME=usecases/verify_claim`.
"""

from mimir import Context, Decider, Mimir, Passage, VerifyResult


def check(decider: Decider, claim: str, evidence: list[str]) -> VerifyResult:
    """Judge one claim against its evidence passages."""
    context = Context(passages=tuple(Passage(text=text) for text in evidence))
    return decider.verify(context, claim, risk=0.01)


def main() -> None:
    model = Mimir.from_pretrained("Mythologic/MIMIR-1")
    result = check(
        model,
        "The export failed every night this week.",
        ["The export has failed every night this week.", "No failures were seen in March."],
    )
    print(result.status, result.answer)


if __name__ == "__main__":
    main()
