import pytest
from one_hot_encoder import fit_transform


def test_empty_input():
    with pytest.raises(TypeError):
        fit_transform()


def test_invalid_input():
    with pytest.raises(TypeError):
        fit_transform(0)


def test_check_the_order():
    result = fit_transform('check', 'the', 'order')
    expected = [
        ('check', [0, 0, 1]),
        ('the', [0, 1, 0]),
        ('order', [1, 0, 0])
    ]
    assert result == expected


def test_list_check_the_order():
    result = fit_transform(['check', 'the', 'order'])
    assert ('check', [0, 0, 1]) in result


def test_check_the_check():
    result = fit_transform('check', 'the', 'check')
    expected = [
        ('check', [0, 1]),
        ('the', [1, 0]),
        ('check', [0, 1])
    ]
    assert result == expected
