CC = gcc
CFLAGS = -fPIC -Wall -Wextra -O2 -g -I/usr/local/cuda/include -pthread
LDFLAGS = -shared -ldl -L/usr/local/cuda/lib64 -lcudart -pthread

# Output
TARGET = libcuda_interceptor.so
INSTALL_DIR = ../../native

# Source files
SOURCES = allocator.c \
          interceptor.c \
          snapshot.c \
          api.c

OBJECTS = $(SOURCES:.c=.o)
HEADERS = types.h config.h allocator.h interceptor.h snapshot.h

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "Build complete: $(TARGET)"

%.o: %.c $(HEADERS)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

install: $(TARGET)
	@mkdir -p $(INSTALL_DIR)
	@cp $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"

clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(OBJECTS) $(TARGET)
	@echo "Clean complete"

# Development helpers
.PHONY: debug rebuild

debug: CFLAGS += -DDEBUG -O0
debug: clean all

rebuild: clean all
