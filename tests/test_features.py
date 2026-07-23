import numpy as np

from cropstressfm.features import exact_kmer_frequencies, promoter_views, reverse_complement


def test_exact_kmer_frequencies() -> None:
    values = exact_kmer_frequencies("ACGT", k=2)
    expected = np.zeros(16, dtype=np.float32)
    expected[[1, 6, 11]] = 1.0 / 3.0
    np.testing.assert_allclose(values, expected)


def test_ambiguous_bases_break_kmers() -> None:
    values = exact_kmer_frequencies("AAAAANCCCCC", k=5)
    assert values[0] == 1.0 / 2.0
    assert values[341] == 1.0 / 2.0
    assert values.sum() == 1.0


def test_promoter_views_keep_tss_proximal_suffix() -> None:
    sequence = "A" * 1700 + "C" * 300 + "G" * 200
    full, proximal_200, proximal_500, proximal_2000 = promoter_views(sequence)
    assert len(full) == 2000
    assert proximal_200 == "G" * 200
    assert proximal_500 == "C" * 300 + "G" * 200
    assert proximal_2000 == full


def test_reverse_complement() -> None:
    assert reverse_complement("ACGTN") == "NACGT"
