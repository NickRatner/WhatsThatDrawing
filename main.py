import pygame
import pygame_gui
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
import numpy
from FoodNeuralNetwork import FoodNeuralNetwork


def validColor(value): #returns if a string is a valid color representation (a number between 0 and 255)
    return value.isnumeric() and int(value) >= 0 and int(value) <= 255

def between(a,b,c): #return if a is between b and c
    return a >= b and a <= c

def eraser():
    color[0] = 200
    color[1] = 200
    color[2] = 200

def setColor():
    #if redInput.get_text().isnumeric() and greenInput.get_text().isnumeric() and blueInput.get_text().isnumeric() and between(int(redInput.get_text()), 0, 255) and between(int(greenInput.get_text()), 0, 255) and between(int(blueInput.get_text()), 0, 255):
    if validColor(redInput.get_text()) and validColor(greenInput.get_text()) and validColor(blueInput.get_text()):
            color[0] = int(redInput.get_text())
            color[1] = int(greenInput.get_text())
            color[2] = int(blueInput.get_text())

            redInput.clear()
            greenInput.clear()
            blueInput.clear()

def confirm(): #confirms the current drawing
    print("Confirm function called")

    with torch.no_grad(): # as we are only using the model, we don't need to back-propagate
        drawing = torch.tensor(canvas) # convert cavnas (current drawing) to tensor
        rawPrediction = model(drawing) # predict label based on drawing
        predictionProbabilities = nn.Softmax(dim=1)(rawPrediction) # gets the probability for each class
        prediction = predictionProbabilities.argmax(1) # pick the highest probability class as the final guess

        print(f"Food: {prediction}") # displays prediction


def reset(): #resets the canvas
    for i in range(numberOfPixels):
        for j in range(numberOfPixels):
            canvas[i][j] = (200,200,200)


def drawWindow(window):
    pygame.draw.rect(window, (0, 0, 0), (0, 0, windowSideLength, windowSideLength)) #draws canvas background
    pygame.draw.rect(window, (0, 200, 0), (windowSideLength, 0, sidePanelWidth, windowSideLength)) #draws side panel

    #draws side menu

    pygame.draw.rect(window, (200, 200, 200), (windowSideLength + 37.5, 50, 75, 50)) #draws reset button
    pygame.draw.rect(window, (200, 200, 200), (windowSideLength + 37.5, 150, 75, 50))  #draws confirm button
    pygame.draw.rect(window, (200, 200, 200), (windowSideLength + 37.5, 150, 75, 50))  # draws eraser button

    pygame.draw.rect(window, (200, 200, 200), (windowSideLength + 37.5, 425, 75, 30))  # draws set color button

    #draws pixels
    for i in range(numberOfPixels):
        for j in range(numberOfPixels):
            pygame.draw.rect(window, canvas[i][j], (i * sizeOfPixel + (sizeOfPixel * 0.0125), j * sizeOfPixel + (sizeOfPixel * 0.0125), sizeOfPixel * 0.975, sizeOfPixel * 0.975)) #draws each pixel


    # draw button text
    font = pygame.font.Font('freesansbold.ttf', 15)

    resetButtonText = font.render("Reset", True, (100, 100, 100))
    confirmButtontText = font.render("Confirm", True, (100, 100, 100))
    setColorButtontText = font.render("Set Color", True, (100, 100, 100))
    redButtonText = font.render("R:", True, (100, 100, 100))
    greenButtontText = font.render("G:", True, (100, 100, 100))
    blueButtontText = font.render("B:", True, (100, 100, 100))

    resetButtonTextRect = resetButtonText.get_rect()
    confirmButtonTextRect = confirmButtontText.get_rect()
    setColorButtonTextRect = setColorButtontText.get_rect()
    redButtonTextRect = redButtonText.get_rect()
    greenButtontTextRect = greenButtontText.get_rect()
    blueButtontTextRect = blueButtontText.get_rect()

    resetButtonTextRect.center = (windowSideLength + 75, 75)
    confirmButtonTextRect.center = (windowSideLength + 75, 175)
    setColorButtonTextRect.center = (windowSideLength + 75, 440)
    redButtonTextRect.center = (windowSideLength + 35, 290)
    greenButtontTextRect.center = (windowSideLength + 35, 340)
    blueButtontTextRect.center = (windowSideLength + 35, 390)

    window.blit(resetButtonText, resetButtonTextRect)
    window.blit(confirmButtontText, confirmButtonTextRect)
    window.blit(setColorButtontText, setColorButtonTextRect)
    window.blit(redButtonText, redButtonTextRect)
    window.blit(greenButtontText, greenButtontTextRect)
    window.blit(blueButtontText, blueButtontTextRect)

    window.blit(eraserIcon, (windowSideLength + 75, 225))

# Initialize ML stuff

transform = Compose([
    Resize(256),                # Resize smaller edge to 256 while preserving aspect ratio
    CenterCrop(224),            # Crop to 224x224
    ToTensor(),                 # Convert image to tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing to ImageNet stats
])

# Download training data from open datasets.
training_data = datasets.Food101(
    root="data",
    split="train",
    download=False,
    transform=transform,
)

# Download test data from open datasets.
test_data = datasets.Food101(
    root="data",
    split="test",
    download=False,
    transform=transform,
)

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 5

# Create data loaders (load training and testing data).
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# Define device on which to run model
device = "cpu"

# Create Model
model = FoodNeuralNetwork().to(device)


# Define Loss Function and Optimizer
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Evaluate Model Functiom
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Train and Evaluate Model

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, lossFunction, optimizer)
    test(test_dataloader, model, lossFunction)
print("Done!")

# Save Model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")



color = [0,0,0]
windowSideLength = 500 #window is a square so this is both length and width
sidePanelWidth = 150
numberOfPixels = 25 #the number of pixels in each dimension (ex. 28 means the screen is 28x28 pixels)
sizeOfPixel = windowSideLength / numberOfPixels

canvas = [[(200,200,200)] * numberOfPixels for _ in range(numberOfPixels)] #create array to store drawing

pygame.init()
window = pygame.display.set_mode((windowSideLength + sidePanelWidth, windowSideLength))
pygame.display.set_caption("Whats That Drawing")
eraserIcon = pygame.image.load("eraserIcon.png")

# draws RGB inputs
manager = pygame_gui.UIManager((windowSideLength + sidePanelWidth, windowSideLength))

redInput = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((windowSideLength + 50, 275), (50, 30)), manager=manager, object_id="#redInput")
greenInput = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((windowSideLength + 50, 325), (50, 30)), manager=manager, object_id="#greenInput")
blueInput = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((windowSideLength + 50, 375), (50, 30)), manager=manager, object_id="#blueInput")

# updates gui manager
UI_REFRESH_RATE = pygame.time.Clock().tick(60) / 5000  # sets the rate at which the GUI will refresh
manager.update(UI_REFRESH_RATE)

mouseDrawing = False

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT: #when window is closed
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:  # when user clicks
            mx, my = pygame.mouse.get_pos()

            if mx > windowSideLength + 37.5 and mx < windowSideLength + 112.5 and my > 50 and my < 100: # reset button pressed
                reset()

            if mx > windowSideLength + 37.5 and mx < windowSideLength + 112.5 and my > 150 and my < 200: #confirm button pressed
                confirm()

            if mx > windowSideLength + 37.5 and mx < windowSideLength + 112.5 and my > 425 and my < 455: #set color button pressed
                setColor()

            if mx > windowSideLength + 75 and mx < windowSideLength + 105 and my > 225 and my < 255: #eraser icon is pressed
                eraser()

            if mx < windowSideLength: #if click is on the canvas
                mouseDrawing = True

            if mx > 0 and mx < 500 and my > 0 and my < 500: #check if mouse is within canvas
                canvas[int(mx / sizeOfPixel)][int(my / sizeOfPixel)] = color.copy()


        if mouseDrawing and event.type == pygame.MOUSEMOTION: #if the user is drawing, and moves the mouse
            mx, my = pygame.mouse.get_pos()
            if mx > 0 and mx < 500 and my > 0 and my < 500: #check if mouse is within canvas
                canvas[int(mx / sizeOfPixel)][int(my / sizeOfPixel)] = color.copy()
            else:
                mouseDrawing = False

        if event.type == pygame.MOUSEBUTTONUP and mouseDrawing:
            mouseDrawing = False

        pygame.draw.rect(window, (200, 200, 200), (windowSideLength + 37.5, 150, 75, 50))
        manager.process_events(event)

    drawWindow(window)


    # Update display
    manager.update(UI_REFRESH_RATE)
    manager.draw_ui(window)  # updates gui components (text input field)
    pygame.display.update()  #update all visuals

pygame.display.quit()
