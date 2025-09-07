import copy
import random
import pygame
import joblib
import numpy as np
from scipy.signal import welch

'''
SECTION COMMENTED OUT AS LIVE IMPLEMENTATION FAILED

# Load model and scaler
model = joblib.load("trained_model.pkl")
scaler = joblib.load("scaler.pkl")

# Simulated biosignal input (replace with real-time later)
def get_simulated_biosignal():
    return {
        "ch_1_filtered": np.random.randn(250),  # EMG
        "ch_2_filtered": np.random.randn(250),  # EEG (Fp2)
        "ch_6_filtered": np.random.randn(250),  # EOG
        "ch_7_filtered": np.random.randn(250),  # EEG (O1)
    }

# Extract features from signal buffer
def extract_features_from_buffer(signal_buffer):
    features = []

    # EMG (ch_1_filtered)
    emg_signal = signal_buffer["ch_1_filtered"]
    features.extend([
        np.mean(np.abs(emg_signal)),
        np.std(emg_signal),
        np.max(np.abs(emg_signal)),
        np.sum(np.square(emg_signal)),
        np.count_nonzero(np.diff(np.sign(emg_signal)))
    ])

    # EOG (ch_6_filtered)
    eog_signal = signal_buffer["ch_6_filtered"]
    features.extend([
        np.mean(eog_signal),
        np.std(eog_signal),
        np.ptp(eog_signal),
        np.sum(np.abs(np.diff(eog_signal))),
        np.count_nonzero(np.diff(np.sign(eog_signal)))
    ])

    # EEG (ch_2_filtered and ch_7_filtered)
    for ch in ["ch_2_filtered", "ch_7_filtered"]:
        eeg_signal = signal_buffer[ch]
        freqs, psd = welch(eeg_signal, fs=250)
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 12)])
        beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
        features.extend([
            np.mean(eeg_signal),
            np.std(eeg_signal),
            np.median(eeg_signal),
            alpha_power,
            beta_power
        ])

    return features
'''
# Initialise Pygame and set up window
pygame.init()
WIDTH, HEIGHT = 800, 500
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption('Water Sort - Biosignal Control')
font = pygame.font.Font('freesansbold.ttf', 24)
clock = pygame.time.Clock()
fps = 60

# Define bubble colours (RGB values)
colour_choices = [
    (255, 0, 0),
    (135, 206, 235),
    (0, 100, 0),
    (255, 105, 180),
]

# Game state variables
tubes = []
tube_rects = []
initial_colours = []
selected = None
keyboard_index = 0
menu_open = False
menu_selected = 0
menu_hover = None
win = False
predicted_label = ""

# Generate random starting arrangement of tubes
def generate_start():
    num_tubes = random.randint(5, 6)
    colours = [[] for _ in range(num_tubes)]
    available_colours = []
    # Fill colour pool
    for i in range(num_tubes - 2):
        for _ in range(4):
            available_colours.append(i)
    # Assign colours randomly to tubes
    for i in range(num_tubes - 2):
        for _ in range(4):
            chosen = random.choice(available_colours)
            colours[i].append(chosen)
            available_colours.remove(chosen)
    return num_tubes, colours

# Draw all tubes and bubbles on screen
def draw_tubes(tube_count, tube_colours):
    tube_boxes = []
    spacing = WIDTH // (tube_count + 1)
    # Tube positioning and dimensions
    for i in range(tube_count):
        x_pos = (i + 1) * spacing
        y_pos = 100
        width, height = 60, 200
        gap_top = 10
        bubble_height = (height - 2 * gap_top) // 4
        tube_rect = pygame.Rect(x_pos - width//2, y_pos, width, height)

        # Draw tube background and outline
        pygame.draw.rect(screen, (200, 200, 200), tube_rect)
        pygame.draw.rect(screen, (0, 0, 0), tube_rect, 3)

        # Highlight selected or keyboard-focused tube
        if i == keyboard_index:
            pygame.draw.rect(screen, (0, 0, 255), tube_rect, 4)
        if selected == i:
            pygame.draw.rect(screen, (0, 200, 0), tube_rect, 5)

        # Draw bubbles inside the tube
        for j, colour_index in enumerate(reversed(tube_colours[i])):
            bubble_rect = pygame.Rect(
                x_pos - width//2 + 5,
                y_pos + gap_top + j * bubble_height,
                width - 10,
                bubble_height - 4
            )
            pygame.draw.rect(screen, colour_choices[colour_index], bubble_rect)
            pygame.draw.rect(screen, (0, 0, 0), bubble_rect, 2)

        tube_boxes.append(tube_rect)
    return tube_boxes

# Move bubble chain from one tube to another if valid
def calc_move(tube_colours, src, dest):
    if src == dest or not tube_colours[src]:
        return tube_colours

    moving_colour = tube_colours[src][-1]
    move_chain = 1

    # Count consecutive bubbles of same colour
    for i in range(len(tube_colours[src]) - 2, -1, -1):
        if tube_colours[src][i] == moving_colour:
            move_chain += 1
        else:
            break

    # Only move if destination has space and colour matches (or is empty)
    if len(tube_colours[dest]) >= 4:
        return tube_colours
    if not tube_colours[dest] or tube_colours[dest][-1] == moving_colour:
        for _ in range(move_chain):
            if len(tube_colours[dest]) < 4:
                tube_colours[dest].append(tube_colours[src].pop())
    return tube_colours

# Check if all tubes are solved (same colour or empty)
def check_victory(tube_colours):
    for tube in tube_colours:
        if len(tube) == 0:
            continue
        if len(tube) != 4:
            return False
        if not all(col == tube[0] for col in tube):
            return False
    return True

# Draw pause menu overlay with buttons
def draw_menu(selected_idx):
    labels = ['Continue', 'New Game', 'Restart']
    menu_height = 80
    menu_width = 600
    menu_x = WIDTH // 2 - menu_width // 2
    menu_y = HEIGHT // 2 - menu_height // 2
    btn_width = 180
    btn_height = 50
    btn_spacing = 30

    # Dim background
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(180)
    overlay.fill((50, 50, 50))
    screen.blit(overlay, (0, 0))

    # Draw menu buttons
    for i, label in enumerate(labels):
        rect = pygame.Rect(menu_x + i * (btn_width + btn_spacing), menu_y, btn_width, btn_height)
        color = (100, 100, 250) if i != selected_idx else (0, 180, 0)
        pygame.draw.rect(screen, color, rect, border_radius=8)
        pygame.draw.rect(screen, (0,0,0), rect, 2, border_radius=8)

        # Draw button labels
        text_surf = font.render(label, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

# Game start setup 
tube_count, tubes = generate_start()
initial_colours = copy.deepcopy(tubes)
running = True

# Main game loop
while running:
    screen.fill((255, 255, 255))
    clock.tick(fps)

    if not menu_open:
        # Draw tubes and check win condition
        tube_rects = draw_tubes(tube_count, tubes)
        win = check_victory(tubes)

        # Display predicted action label (simulated)
        label_text = font.render(f"Prediction: {predicted_label}", True, (0, 0, 0))
        screen.blit(label_text, (10, 10))

        # Display win or instructions
        if win:
            message = font.render('You Win! Press ESC for Menu', True, (0, 150, 0))
            screen.blit(message, (WIDTH//2 - message.get_width()//2, HEIGHT - 50))
        else:
            instructions = font.render('Control with brain + muscle! Simulating...', True, (0, 0, 0))
            screen.blit(instructions, (WIDTH//2 - instructions.get_width()//2, HEIGHT - 30))
    else:
        # Draw pause menu if open
        draw_menu(menu_selected)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Close game if window closed 
            running = False

        elif event.type == pygame.KEYDOWN:
            if not menu_open:
                # Game controls simulated by keyboard
                if event.key == pygame.K_ESCAPE:
                    print("Prediction: mental_relax")
                    menu_open = True
                # Move left
                elif event.key == pygame.K_LEFT:
                    print("Prediction: look_left")
                    predicted_label = "look_left"
                    keyboard_index = (keyboard_index - 1) % tube_count
                # Move right
                elif event.key == pygame.K_RIGHT:
                    print("Prediction: look_right")
                    predicted_label = "look_right"
                    keyboard_index = (keyboard_index + 1) % tube_count
                # Selecting 
                elif event.key == pygame.K_SPACE:
                    print("Prediction: jaw_clench")
                    predicted_label = "jaw_clench"
                    if selected is None and tubes[keyboard_index]:
                        selected = keyboard_index
                    else:
                        tubes = calc_move(tubes, selected, keyboard_index)
                        selected = None
                # Open menu via relax
                elif event.key == pygame.K_m:
                    print("Prediction: mental_relax")
                    predicted_label = "mental_relax"
                    menu_open = True
            else:
                # Menu navigation controls
                if event.key == pygame.K_LEFT:
                    print("Prediction: look_left")
                    menu_selected = (menu_selected - 1) % 3
                elif event.key == pygame.K_RIGHT:
                    print("Prediction: look_right")
                    menu_selected = (menu_selected + 1) % 3
                elif event.key == pygame.K_SPACE:
                    # Continue game
                    if menu_selected == 0:  
                        menu_open = False
                    # New game    
                    elif menu_selected == 1: 
                        tube_count, tubes = generate_start()
                        initial_colours = copy.deepcopy(tubes)
                        selected = None
                        keyboard_index = 0
                        menu_open = False
                        win = False
                    # Restart current game
                    elif menu_selected == 2:  
                        tubes = copy.deepcopy(initial_colours)
                        selected = None
                        keyboard_index = 0
                        menu_open = False

    pygame.display.flip()

# Quit Pygame
pygame.quit()
