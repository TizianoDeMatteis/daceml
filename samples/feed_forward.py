import torch
import torch.nn as nn
import sys
# sys.path.append('../')
import dace
from dace.transformation.interstate import FPGATransformSDFG

from daceml.pytorch import DaceModule

class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output


module = Feedforward(5,5)
res = module(torch.ones(5))
print(res)

dace_module = DaceModule(module)
res2 = dace_module(torch.ones(5))
print("Computed: ", res2[0])
print("Expected: ", res.item())
if abs(res2[0] - res.item())< 1e-5:
    print("Result is correct")
else:
    print("Result is WRONG")
sdfg = dace_module.sdfg
sdfg.save('/tmp/out.sdfg')
sdfg.apply_transformations([FPGATransformSDFG])
sdfg.states()[0].location["is_FPGA_kernel"]=False
sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"]=False
sdfg.save('/tmp/out_fpga.sdfg')

# import pdb
# pdb.set_trace()
sdfg.expand_library_nodes()
sdfg.save('/tmp/out_fpga_expanded.sdfg')

res3 = dace_module(torch.ones(5))
if abs(res3[0] - res.item())< 1e-5:
    print("Result is correct")
else:
    print("Result is WRONG")
