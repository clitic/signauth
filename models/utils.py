import torch


class ptjit:
	"""
	load pytorch models without model class code

	Usage
	-----

	# Basic Requirements
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ModelClass(*args, **kwargs)

	# Saving
	ptjit("modeljit.pth").save(model, torch.randn(1, 3, 256, 256))

	# Loading
	model = ptjit("modeljit.pth").load().to(device)
	with torch.no_grad():
		print(model(torch.randn(1, 3, 256, 256)))
	"""

	def __init__(self, path):
		self.path = path

	def save(self, modelclass, standardinput):
		traced_cell = torch.jit.trace(modelclass, standardinput)
		torch.jit.save(traced_cell, self.path)

	def load(self):
		return torch.jit.load(self.path)

