import unittest

import torch

from src.losses import BCEDiceLoss, TverskyDiceLoss, FocalDiceLoss


class MaskedLossTests(unittest.TestCase):
    def setUp(self):
        self.losses = (
            BCEDiceLoss(),
            TverskyDiceLoss(),
            FocalDiceLoss(),
        )

    def check_finite_backward(self, target, ignore):
        for criterion in self.losses:
            logits = torch.tensor(
                [[[[0.2, -0.4], [1.0, -1.0]]]],
                dtype=torch.float32,
                requires_grad=True,
            )
            loss = criterion(logits, target, ignore)
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_normal_labelled_batch(self):
        target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        ignore = torch.zeros_like(target, dtype=torch.bool)
        self.check_finite_backward(target, ignore)

    def test_partially_ignored_batch_excludes_ignored_pixels(self):
        target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        ignore = torch.tensor([[[False, True], [False, False]]])
        self.check_finite_backward(target, ignore)
        # Changing only an ignored target cannot change any masked loss.
        altered = target.clone()
        altered[ignore] = 99.0
        logits = torch.zeros((1, 1, 2, 2))
        for criterion in self.losses:
            self.assertTrue(torch.allclose(
                criterion(logits, target, ignore),
                criterion(logits, altered, ignore),
            ))

    def test_completely_ignored_batch_has_finite_zero_gradients(self):
        target = torch.zeros((1, 2, 2))
        ignore = torch.ones_like(target, dtype=torch.bool)
        for criterion in self.losses:
            logits = torch.randn((1, 1, 2, 2), requires_grad=True)
            loss = criterion(logits, target, ignore)
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all().item())
            self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_expected_shapes_and_invalid_shapes(self):
        logits = torch.zeros((2, 1, 3, 4))
        targets = torch.zeros((2, 3, 4))
        ignore = torch.zeros((2, 3, 4), dtype=torch.bool)
        for criterion in self.losses:
            self.assertTrue(torch.isfinite(criterion(logits, targets, ignore)))
            with self.assertRaises(ValueError):
                criterion(logits, targets.unsqueeze(1), ignore)
            with self.assertRaises(ValueError):
                criterion(logits, targets, ignore[:, :, :-1])


if __name__ == "__main__":
    unittest.main()
