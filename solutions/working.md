
### GPU Working


```
Grid
└── Block 0
    ├── Thread 0
    ├── Thread 1
    ├── Thread 2
    └── Thread 3
```



kernelGPUFunction<<<numBlocks, threadsPerBlock, sharedMemBytes, stream>>>(args);



CUDA memory is:
* Allocated via ioctl to /dev/nvidiactl
* Mapped into GPU virtual address space
* Managed by UVM + driver
* Invisible to /proc/<pid>/maps



Process
 ├── CPU VA (kernel-managed)
 └── GPU VA (driver-managed, opaque)
