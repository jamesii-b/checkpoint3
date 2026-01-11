import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam

from src.runtime.training_runtime import TrainingRuntime, TrainingState


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_training_runtime_basic():
    runtime = TrainingRuntime()
    
    model = SimpleModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    runtime.register_model(model)
    runtime.register_optimizer(optimizer)
    runtime.set_epoch(5)
    runtime.set_step(100)
    runtime.set_loss(0.25)
    
    state = runtime.get_training_state()
    
    assert state.epoch == 5
    assert state.step == 100
    assert state.loss == 0.25
    assert state.lr == 0.001
    assert "fc1.weight" in state.model_state
    assert "fc2.bias" in state.model_state


def test_training_state_with_gradients():
    runtime = TrainingRuntime()
    
    model = SimpleModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(4, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    runtime.register_model(model)
    runtime.register_optimizer(optimizer)
    
    state = runtime.get_training_state()
    
    assert state.grad_state["fc1.weight"] is not None
    assert state.grad_state["fc2.weight"] is not None


def test_serialize_deserialize():
    runtime = TrainingRuntime()
    
    model = SimpleModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(4, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    runtime.register_model(model)
    runtime.register_optimizer(optimizer)
    runtime.set_epoch(10)
    runtime.set_step(500)
    runtime.set_loss(0.123)
    
    state = runtime.get_training_state()
    serialized = runtime.serialize_state(state)
    
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0
    
    restored_state = runtime.deserialize_state(serialized)
    
    assert restored_state.epoch == 10
    assert restored_state.step == 500
    assert abs(restored_state.loss - 0.123) < 1e-6
    
    for name in state.model_state:
        assert torch.allclose(state.model_state[name], restored_state.model_state[name])


def test_restore_training_state():
    model1 = SimpleModel()
    optimizer1 = Adam(model1.parameters(), lr=0.001)
    runtime1 = TrainingRuntime()
    runtime1.register_model(model1)
    runtime1.register_optimizer(optimizer1)
    
    for _ in range(5):
        x = torch.randn(4, 10)
        y = model1(x)
        loss = y.sum()
        loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()
    
    runtime1.set_epoch(5)
    runtime1.set_step(50)
    state = runtime1.get_training_state()
    serialized = runtime1.serialize_state(state)
    
    model2 = SimpleModel()
    optimizer2 = Adam(model2.parameters(), lr=0.001)
    runtime2 = TrainingRuntime()
    runtime2.register_model(model2)
    runtime2.register_optimizer(optimizer2)
    
    restored_state = runtime2.deserialize_state(serialized)
    runtime2.restore_training_state(restored_state)
    
    for name, param1 in model1.named_parameters():
        param2 = dict(model2.named_parameters())[name]
        assert torch.allclose(param1, param2)
    
    state1 = optimizer1.state_dict()
    state2 = optimizer2.state_dict()
    
    for param_id in state1['state']:
        for key in state1['state'][param_id]:
            if isinstance(state1['state'][param_id][key], torch.Tensor):
                assert torch.allclose(
                    state1['state'][param_id][key],
                    state2['state'][param_id][key]
                )


def test_optimizer_momentum_preserved():
    model = SimpleModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    runtime = TrainingRuntime()
    runtime.register_model(model)
    runtime.register_optimizer(optimizer)
    
    for _ in range(10):
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    state = runtime.get_training_state()
    
    assert 'state' in state.optimizer_state
    assert len(state.optimizer_state['state']) > 0
    
    for param_state in state.optimizer_state['state'].values():
        assert 'exp_avg' in param_state
        assert 'exp_avg_sq' in param_state


if __name__ == "__main__":
    test_training_runtime_basic()
    print("test_training_runtime_basic passed")
    
    test_training_state_with_gradients()
    print("test_training_state_with_gradients passed")
    
    test_serialize_deserialize()
    print("test_serialize_deserialize passed")
    
    test_restore_training_state()
    print("test_restore_training_state passed")
    
    test_optimizer_momentum_preserved()
    print("test_optimizer_momentum_preserved passed")
    
    print("\nAll training_runtime tests passed!")
