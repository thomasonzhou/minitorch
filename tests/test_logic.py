import minitorch as torch


def test_where():
    shape = (2, 3)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    t1 = torch.ones(*shape)
    t2 = torch.zeros(shape)

    t3 = torch.where(mask, t1, t2)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if mask[i, j]:
                assert t3[i, j] == t1[i, j]
            else:
                assert t3[i, j] == t2[i, j]
