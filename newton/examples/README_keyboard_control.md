# Robot Keyboard Controller

A reusable keyboard control system for robot simulations using pygame.

## Features

- **Real-time keyboard input** with visual feedback window
- **Customizable command dimensions** (2D, 3D, or higher)
- **Configurable key mappings** for different robot types
- **Command value limiting** and smooth incremental control
- **Cross-robot compatibility** through modular design

## Installation

```bash
pip install pygame torch numpy
```

## Quick Start

### Basic Usage

```python
from robot_keyboard_controller import RobotKeyboardController

# Create a basic controller
controller = RobotKeyboardController(device="cpu")

try:
    while True:
        # Update returns False when user wants to quit
        if not controller.update(verbose=True):
            break
            
        # Get current command tensor
        command = controller.get_command()
        print(f"Command: {command}")
        
        # Use command to control your robot here
        
finally:
    controller.cleanup()
```

### G1 Robot Example

```python
from robot_keyboard_controller import create_g1_controller

# Create G1-specific controller
controller = create_g1_controller(device="cuda")

# In your simulation loop:
running = controller.update(verbose=True)
robot_command = controller.get_command()  # Shape: (1, 3)
# robot_command[0, 0] = forward/backward
# robot_command[0, 1] = left/right  
# robot_command[0, 2] = rotation (if using Q/E keys)
```

## Configuration Options

### Constructor Parameters

- `device`: PyTorch device ("cpu" or "cuda")
- `command_size`: Number of command dimensions (default: 3)
- `step_size`: Increment step size (default: 0.05)
- `command_limits`: Min/max values tuple (default: (-1.0, 1.0))
- `window_size`: pygame window dimensions (default: (500, 400))
- `window_title`: Window title string

### Custom Robot Example

```python
from robot_keyboard_controller import create_custom_controller

# Create custom controller for a 2D robot
controller = create_custom_controller(
    device="cpu",
    command_size=2,           # Only X, Y movement
    step_size=0.1,            # Larger steps
    limits=(-2.0, 2.0),       # Extended range
    title="My Robot Control"
)
```

## Key Mappings

### Default Controls
- **↑/↓ Arrow Keys**: Forward/Backward (command index 0)
- **←/→ Arrow Keys**: Left/Right (command index 1)  
- **Q/E Keys**: Rotate Left/Right (command index 2)
- **Spacebar**: Reset all commands to zero
- **Close Window**: Exit program

### Custom Key Mappings

```python
import pygame

custom_mappings = {
    pygame.K_w: ("forward", 0, 1),
    pygame.K_s: ("backward", 0, -1),
    pygame.K_a: ("left", 1, 1),
    pygame.K_d: ("right", 1, -1),
    pygame.K_SPACE: ("reset", -1, 0),
}

controller.configure_key_mapping(custom_mappings)
```

## Integration with Existing Robots

### Step 1: Import the Controller

```python
from robot_keyboard_controller import create_g1_controller
# or
from robot_keyboard_controller import RobotKeyboardController
```

### Step 2: Replace Existing Keyboard Code

Replace your existing pygame initialization and keyboard handling with:

```python
# Old way:
# pygame.init()
# screen = pygame.display.set_mode((400, 300))
# # ... manual key handling ...

# New way:
keyboard_controller = create_g1_controller(device=your_device)
```

### Step 3: Update Your Simulation Loop

```python
try:
    for frame in range(num_frames):
        # Handle keyboard input
        if not keyboard_controller.update(verbose=True):
            break
            
        # Update robot command
        robot.command = keyboard_controller.get_command()
        
        # Continue with simulation
        robot.step()
        robot.render()
        
finally:
    keyboard_controller.cleanup()
```

## Advanced Features

### Context Manager Support

```python
with RobotKeyboardController() as controller:
    while controller.update():
        command = controller.get_command()
        # Use command...
# Automatic cleanup
```

### Command Manipulation

```python
# Get current command
current_cmd = controller.get_command()

# Set command directly  
new_cmd = torch.tensor([[0.5, -0.3, 0.0]])
controller.set_command(new_cmd)

# Reset to zero
controller.reset_commands()
```

## Examples

See `keyboard_controller_example.py` for complete examples including:
- Basic usage
- G1 robot control  
- Custom 2D robot
- Advanced configuration with custom keys

## Troubleshooting

### Import Errors
Make sure pygame is installed: `pip install pygame`

### Display Issues
- Ensure you have a display available (not running in headless mode)
- Try different pygame display drivers if needed

### Command Not Updating
- Check that `controller.update()` is being called in your main loop
- Verify the pygame window has focus for key input

## File Structure

- `robot_keyboard_controller.py` - Main controller implementation
- `example_g1_policy.py` - Updated G1 example using the controller
- `keyboard_controller_example.py` - Standalone examples
- `requirements_example.txt` - Required dependencies 