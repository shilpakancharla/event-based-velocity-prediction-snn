"""
  Custom collation function for event-based multi-output problems. 
  Adapted from https://github.com/neuromorphs/tonic/blob/develop/tonic/collation.py.
"""

class PadMultiOutputTensor:
  def __init__(self, batch_first: bool = False):
    self.batch_first = batch_first

  def __call__(self, batch):
    samples_output = []
    targets_output = []

    max_length = max([sample.shape[0] for sample, target in batch])
    for sample, target in batch:
      if isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample)
        samples_output.append(
            torch.cat((sample,
                       torch.zeros(max_length - sample.shape[0], *sample.shape[1:]),)))
        targets_output.append(target)
        print(targets_output)
    return (
        torch.stack(samples_output, 0 if self.batch_first else 1),
        torch.stack(targets_output),
    )
