from research.validation import walk_forward


def test_first_window_slices():
    n = 100
    train = 50
    test = 10
    it = walk_forward(n, train, test)
    tr_s, te_s = next(it)
    assert (tr_s.start, tr_s.stop) == (0, train)
    assert (te_s.start, te_s.stop) == (train, train + test)

