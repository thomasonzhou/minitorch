import minitorch as torch


def test_tensor():
    l1 = [1, 1, 2, 3, 5, 8]
    t1 = torch.tensor(l1)
    assert len(l1) == len(t1._tensor._storage)

    for i in range(len(l1)):
        assert torch.isclose(t1[i], l1[i])
