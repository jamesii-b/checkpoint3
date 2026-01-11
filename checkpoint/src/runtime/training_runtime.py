import io
import struct
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch
from torch.optim import Adam


@dataclass
class TrainingState:
    epoch: int
    step: int
    loss: float
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    grad_state: Dict[str, Optional[torch.Tensor]]
    lr: float


class TrainingRuntime:
    def __init__(self):
        self._model = None
        self._optimizer = None
        self._epoch = 0
        self._step = 0
        self._loss = 0.0

    def register_model(self, model: torch.nn.Module) -> None:
        self._model = model

    def register_optimizer(self, optimizer: Adam) -> None:
        self._optimizer = optimizer

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def set_step(self, step: int) -> None:
        self._step = step

    def set_loss(self, loss: float) -> None:
        self._loss = loss

    def get_training_state(self) -> TrainingState:
        model_state = {}
        grad_state = {}

        if self._model:
            for name, param in self._model.named_parameters():
                model_state[name] = param.data.clone()
                grad_state[name] = param.grad.clone() if param.grad is not None else None

        optimizer_state = {}
        if self._optimizer:
            optimizer_state = self._optimizer.state_dict()

        return TrainingState(
            epoch=self._epoch,
            step=self._step,
            loss=self._loss,
            model_state=model_state,
            optimizer_state=optimizer_state,
            grad_state=grad_state,
            lr=self._optimizer.param_groups[0]["lr"] if self._optimizer else 0.0,
        )

    def restore_training_state(self, state: TrainingState) -> None:
        self._epoch = state.epoch
        self._step = state.step
        self._loss = state.loss

        if self._model and state.model_state:
            for name, param in self._model.named_parameters():
                if name in state.model_state:
                    param.data.copy_(state.model_state[name])
                if name in state.grad_state and state.grad_state[name] is not None:
                    if param.grad is None:
                        param.grad = state.grad_state[name].clone()
                    else:
                        param.grad.copy_(state.grad_state[name])

        if self._optimizer and state.optimizer_state:
            self._optimizer.load_state_dict(state.optimizer_state)

    def serialize_state(self, state: TrainingState) -> bytes:
        buffer = io.BytesIO()

        buffer.write(struct.pack("<I", state.epoch))
        buffer.write(struct.pack("<Q", state.step))
        buffer.write(struct.pack("<d", state.loss))
        buffer.write(struct.pack("<d", state.lr))

        model_buffer = io.BytesIO()
        torch.save(state.model_state, model_buffer)
        model_bytes = model_buffer.getvalue()
        buffer.write(struct.pack("<Q", len(model_bytes)))
        buffer.write(model_bytes)

        opt_buffer = io.BytesIO()
        torch.save(state.optimizer_state, opt_buffer)
        opt_bytes = opt_buffer.getvalue()
        buffer.write(struct.pack("<Q", len(opt_bytes)))
        buffer.write(opt_bytes)

        grad_buffer = io.BytesIO()
        torch.save(state.grad_state, grad_buffer)
        grad_bytes = grad_buffer.getvalue()
        buffer.write(struct.pack("<Q", len(grad_bytes)))
        buffer.write(grad_bytes)

        return buffer.getvalue()

    def deserialize_state(self, data: bytes) -> TrainingState:
        buffer = io.BytesIO(data)

        epoch = struct.unpack("<I", buffer.read(4))[0]
        step = struct.unpack("<Q", buffer.read(8))[0]
        loss = struct.unpack("<d", buffer.read(8))[0]
        lr = struct.unpack("<d", buffer.read(8))[0]

        model_len = struct.unpack("<Q", buffer.read(8))[0]
        model_bytes = buffer.read(model_len)
        model_buffer = io.BytesIO(model_bytes)
        model_state = torch.load(model_buffer, weights_only=False)

        opt_len = struct.unpack("<Q", buffer.read(8))[0]
        opt_bytes = buffer.read(opt_len)
        opt_buffer = io.BytesIO(opt_bytes)
        optimizer_state = torch.load(opt_buffer, weights_only=False)

        grad_len = struct.unpack("<Q", buffer.read(8))[0]
        grad_bytes = buffer.read(grad_len)
        grad_buffer = io.BytesIO(grad_bytes)
        grad_state = torch.load(grad_buffer, weights_only=False)

        return TrainingState(
            epoch=epoch,
            step=step,
            loss=loss,
            model_state=model_state,
            optimizer_state=optimizer_state,
            grad_state=grad_state,
            lr=lr,
        )


_training_runtime: Optional[TrainingRuntime] = None


def get_training_runtime() -> TrainingRuntime:
    global _training_runtime
    if _training_runtime is None:
        _training_runtime = TrainingRuntime()
    return _training_runtime
