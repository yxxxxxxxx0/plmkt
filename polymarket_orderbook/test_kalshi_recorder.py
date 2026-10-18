"""
Offline checks for kalshi_recorder's book conversion and diffing -- the two
pieces that decide what gets written. No network.

    python test_kalshi_recorder.py
"""
from kalshi_recorder import BookState, parse_book


def test_parse_book_yes_terms():
    ob = {"yes_dollars": [["0.4200", "10"], ["0.4400", "5.5"], ["0.4300", "0"]],
          "no_dollars": [["0.5000", "7"], ["0.5500", "3"]]}
    bids, asks = parse_book(ob)
    assert bids == ((0.44, 5.5), (0.42, 10.0)), bids          # high -> low, zero size dropped
    assert asks == ((0.45, 3.0), (0.5, 7.0)), asks            # NO bid p -> YES ask 1-p, low -> high


def test_parse_book_empty():
    assert parse_book({}) == ((), ())
    assert parse_book({"yes_dollars": None, "no_dollars": []}) == ((), ())


def test_diff_seed_unchanged_change():
    st = BookState()
    a = (((0.44, 1.0),), ((0.45, 2.0),))
    b = (((0.44, 1.0),), ((0.46, 2.0),))
    assert st.apply("T", *a) == "seed"
    assert st.apply("T", *a) is None
    assert st.apply("T", *b) == "change"


def test_tickers_independent():
    st = BookState()
    assert st.apply("A", (), ()) == "seed"
    assert st.apply("B", (), ()) == "seed"
    assert st.apply("A", (), ()) is None


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed")
