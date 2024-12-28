import minitorch as torch


def test_tensor_create():
    l1 = [1, 1, 2, 3, 5, 8]
    t1 = torch.tensor(l1)
    assert len(l1) == len(t1._tensor._storage)

    for i in range(len(l1)):
        assert t1[i] == l1[i]


def test_tensor_indexing_1():
    t1 = torch.arange(5)
    t2 = t1[:]
    t3 = t1[...]
    for i in range(5):
        assert t1[i] == t2[i] == t3[i]


def test_tensor_indexing_2():
    t1 = torch.arange(6).view(2, 3)
    t2 = t1[:]
    t3 = t1[...]
    t4 = t1[:, ...]
    t5 = t1[..., :]
    t6 = t1[:, :]
    t7 = t1[:2, :3]
    for i in range(2):
        for j in range(3):
            val = i * 3 + j
            assert (
                t1[i, j]
                == t2[i, j]
                == t3[i, j]
                == t4[i, j]
                == t5[i, j]
                == t6[i, j]
                == t7[i, j]
                == val
            )


def test_newaxis():
    t1 = torch.ones(9).view(3, 3)
    t2 = t1[..., None]
    t3 = t1[:, :, torch.newaxis]
    assert t2.shape == t3.shape == (3, 3, 1)


def test_where():
    shape = (2, 3)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    t1 = torch.ones(*shape)
    t2 = torch.zeros(*shape)

    t3 = torch.where(mask, t1, t2)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if mask[i, j]:
                assert t3[i, j] == t1[i, j]
            else:
                assert t3[i, j] == t2[i, j]
