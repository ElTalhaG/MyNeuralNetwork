import pygame
import numpy as np
from PIL import Image
import io

class DrawingBoard:
    def __init__(self, width=280, height=280, background_color=(255, 255, 255), drawing_color=(0, 0, 0)):
        pygame.init()
        
        # Set up the display
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width + 200, height))  # Extra space for buttons
        pygame.display.set_caption("Draw a digit (0-9)")
        
        # Colors
        self.background_color = background_color
        self.drawing_color = drawing_color
        
        # Drawing surface
        self.drawing_surface = pygame.Surface((width, height))
        self.drawing_surface.fill(background_color)
        
        # Drawing properties
        self.drawing = False
        self.last_pos = None
        self.line_width = 15
        
        # Create buttons
        self.font = pygame.font.Font(None, 36)
        self.predict_button = pygame.Rect(width + 20, 50, 160, 50)
        self.clear_button = pygame.Rect(width + 20, 120, 160, 50)
        
    def get_input_image(self):
        """Convert the drawing surface to a format suitable for the neural network"""
        # Create a copy of the surface
        surf_copy = self.drawing_surface.copy()
        
        # Get raw pixel data
        pixel_data = pygame.image.tostring(surf_copy, 'RGB')
        
        # Convert to PIL Image
        img = Image.frombytes('RGB', (self.width, self.height), pixel_data)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 28x28 (MNIST format)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.reshape(784, 1)
        img_array = img_array.astype('float32') / 255.0
        
        # Invert colors (MNIST has white digits on black background)
        img_array = 1 - img_array
        
        return img_array
    
    def run(self, model):
        """Run the drawing board"""
        running = True
        prediction = None
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if clicked on buttons
                    mouse_pos = event.pos
                    if self.predict_button.collidepoint(mouse_pos):
                        # Get prediction
                        img_array = self.get_input_image()
                        prediction = model.predict(img_array)[0]
                    elif self.clear_button.collidepoint(mouse_pos):
                        # Clear the drawing surface
                        self.drawing_surface.fill(self.background_color)
                        prediction = None
                    else:
                        # Start drawing
                        self.drawing = True
                        self.last_pos = mouse_pos
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.drawing = False
                    self.last_pos = None
                
                elif event.type == pygame.MOUSEMOTION and self.drawing:
                    # Draw line from last position to current position
                    mouse_pos = event.pos
                    if mouse_pos[0] < self.width:  # Only draw within the drawing surface
                        if self.last_pos:
                            pygame.draw.line(
                                self.drawing_surface,
                                self.drawing_color,
                                (self.last_pos[0], self.last_pos[1]),
                                (mouse_pos[0], mouse_pos[1]),
                                self.line_width
                            )
                        self.last_pos = mouse_pos
            
            # Clear the screen
            self.screen.fill((200, 200, 200))
            
            # Draw the drawing surface
            self.screen.blit(self.drawing_surface, (0, 0))
            
            # Draw buttons
            pygame.draw.rect(self.screen, (100, 200, 100), self.predict_button)
            pygame.draw.rect(self.screen, (200, 100, 100), self.clear_button)
            
            # Draw button text
            predict_text = self.font.render("Predict", True, (0, 0, 0))
            clear_text = self.font.render("Clear", True, (0, 0, 0))
            self.screen.blit(predict_text, (self.width + 50, 65))
            self.screen.blit(clear_text, (self.width + 60, 135))
            
            # Draw prediction
            if prediction is not None:
                pred_text = self.font.render(f"Prediction: {prediction}", True, (0, 0, 0))
                self.screen.blit(pred_text, (self.width + 20, 200))
            
            pygame.display.flip()
        
        pygame.quit() 