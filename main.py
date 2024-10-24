import pygame
import pygame_gui
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop, Normalize, RandomHorizontalFlip
import numpy
from FoodNeuralNetwork import FoodNeuralNetwork
from FoodConvNeuralNetwork import FoodConvNeuralNetwork
import os


foodLabels = [
    "Apple pie", "Baby back ribs", "Baklava", "Beef carpaccio", "Beef tartare", "Beet salad",
    "Beignets", "Bibimbap", "Bread pudding", "Breakfast burrito", "Bruschetta", "Caesar salad",
    "Cannoli", "Caprese salad", "Carrot cake", "Ceviche", "Cheesecake", "Cheese plate",
    "Chicken curry", "Chicken quesadilla", "Chicken wings", "Chocolate cake", "Chocolate mousse",
    "Churros", "Clam chowder", "Club sandwich", "Crab cakes", "Creme brulee", "Croque madame",
    "Cup cakes", "Deviled eggs", "Donuts", "Dumplings", "Edamame", "Eggs benedict", "Escargots",
    "Falafel", "Filet mignon", "Fish and chips", "Foie gras", "French fries", "French onion soup",
    "French toast", "Fried calamari", "Fried rice", "Frozen yogurt", "Garlic bread", "Gnocchi",
    "Greek salad", "Grilled cheese sandwich", "Grilled salmon", "Guacamole", "Gyoza", "Hamburger",
    "Hot and sour soup", "Hot dog", "Huevos rancheros", "Hummus", "Ice cream", "Lasagna",
    "Lobster bisque", "Lobster roll sandwich", "Macaroni and cheese", "Macarons", "Miso soup",
    "Mussels", "Nachos", "Omelette", "Onion rings", "Oysters", "Pad thai", "Paella", "Pancakes",
    "Panna cotta", "Peking duck", "Pho", "Pizza", "Pork chop", "Poutine", "Prime rib",
    "Pulled pork sandwich", "Ramen", "Ravioli", "Red velvet cake", "Risotto", "Samosa",
    "Sashimi", "Scallops", "Seaweed salad", "Shrimp and grits", "Spaghetti bolognese",
    "Spaghetti carbonara", "Spring rolls", "Steak", "Strawberry shortcake", "Sushi", "Tacos",
    "Takoyaki", "Tiramisu", "Tuna tartare", "Waffles"
]

foodLabelsDict = {i: food for i, food in enumerate(foodLabels)}

def validColor(value): #returns if a string is a valid color representation (a number between 0 and 255)
    return value.isnumeric() and int(value) >= 0 and int(value) <= 255

def between(a,b,c): #return if a is between b and c
    return a >= b and a <= c

def eraser():
    global color
    global usingEraser
    global pickingColor
    color = backgroundColor.copy()
    usingEraser = True
    pickingColor = False

def colorPicker():
    global pickingColor
    global usingEraser
    pickingColor = True
    usingEraser = False

def setColor():
    global usingEraser
    global pickingColor
    if validColor(redInput.get_text()) and validColor(greenInput.get_text()) and validColor(blueInput.get_text()):
            color[0] = int(redInput.get_text())
            color[1] = int(greenInput.get_text())
            color[2] = int(blueInput.get_text())

            redInput.clear()
            greenInput.clear()
            blueInput.clear()

            usingEraser = False
            pickingColor = False

def confirm(): #confirms the current drawing
    canvas_array = numpy.array(canvas)  # This creates a (height, width, 3) NumPy array
    canvas_array = canvas_array / 255.0 # normalize all values

    drawing = torch.tensor(canvas_array,dtype=torch.float32) # Convert current drawing to tensor
    drawing = drawing.permute(2, 0, 1)  # Changes from (height, width, 3) to (3, height, width)

    resize_transform = Resize((224, 224)) # Define resize transformation
    drawing_resized = resize_transform(drawing.unsqueeze(0))  # Add batch dimension and resize, as resize expects to know how many images are being inputted

    normalize_transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    drawing_normalized = normalize_transform(drawing_resized)  # Apply normalization

    with torch.no_grad(): # as we are only using the model, we don't need to back-propagate
        rawPrediction = model(drawing_normalized) # predict label based on drawing
        predictionProbabilities = nn.Softmax(dim=1)(rawPrediction) # gets the probability for each class
        prediction = predictionProbabilities.argmax(1) # pick the highest probability class as the final guess

        print(f"Food: {foodLabelsDict[prediction.item()]}") # displays prediction


def reset(): #resets the canvas
    for i in range(numberOfPixels):
        for j in range(numberOfPixels):
            canvas[i][j] = backgroundColor.copy()


def drawWindow(window):
    pygame.draw.rect(window, (0, 0, 0), (0, 0, windowSideLength, windowSideLength)) #draws canvas background
    pygame.draw.rect(window, (0, 200, 0), (windowSideLength, 0, sidePanelWidth, windowSideLength)) #draws side panel

    #draws side menu

    pygame.draw.rect(window, (200, 200, 200), (windowSideLength + 37.5, 50, 75, 50)) #draws reset button
    pygame.draw.rect(window, (200, 200, 200), (windowSideLength + 37.5, 150, 75, 50))  #draws confirm button
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

    window.blit(eraserIcon, (windowSideLength + 80, 225))
    window.blit(colorPickerIcon, (windowSideLength + 37.5, 225))


# Initialize ML stuff

transform = Compose([
    Resize(256),                # Resize smaller edge to 256 while preserving aspect ratio
    RandomCrop(224),            # Crop to 224x224
    RandomHorizontalFlip(),     # add a random flip to prevent overfitting
    ToTensor(),                 # Convert image to tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing to ImageNet stats
])

# Download training data from open datasets.
training_data = datasets.Food101(
    root="data",
    split="train",
    download=False,
    transform=transform
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
learning_rate = 0.01
epochs = 4

# Create data loaders (load training and testing data).
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Define device on which to run model
device = "cpu"

# Create Model
#model = FoodNeuralNetwork().to(device)
model = FoodConvNeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

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




# Load Model (if already exists)
#if False:
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", weights_only=True))
else:
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
backgroundColor = [200, 200, 200]
windowSideLength = 500 #window is a square so this is both length and width
sidePanelWidth = 150
numberOfPixels = 25 #the number of pixels in each dimension (ex. 28 means the screen is 28x28 pixels)
sizeOfPixel = windowSideLength / numberOfPixels

canvas = [[backgroundColor.copy()] * numberOfPixels for _ in range(numberOfPixels)] #create array to store drawing

pygame.init()
window = pygame.display.set_mode((windowSideLength + sidePanelWidth, windowSideLength))
pygame.display.set_caption("Whats That Drawing")
eraserIcon = pygame.image.load("eraserIcon.png")
colorPickerIcon = pygame.image.load("colorPickerIcon.png")
cursor = pygame.image.load("mouseCursor.png")
colorPickerCursor = pygame.image.load("colorPickerCursor.png")
eraserCursor = pygame.image.load("eraserCursor.png")

pygame.mouse.set_visible(False)
pickingColor = False
usingEraser = False

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

    mx, my = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT: #when window is closed
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:  # when user clicks
            if mx > windowSideLength + 37.5 and mx < windowSideLength + 112.5 and my > 50 and my < 100: # reset button pressed
                reset()

            if mx > windowSideLength + 37.5 and mx < windowSideLength + 112.5 and my > 150 and my < 200: #confirm button pressed
                confirm()

            if mx > windowSideLength + 37.5 and mx < windowSideLength + 112.5 and my > 425 and my < 455: #set color button pressed
                setColor()

            if mx > windowSideLength + 80 and mx < windowSideLength + 110 and my > 225 and my < 255: #eraser icon is pressed
                eraser()

            if mx > windowSideLength + 37.5 and mx < windowSideLength + 67.5 and my > 225 and my < 255: #colorPicker icon is pressed
                colorPicker()


            if mx > 0 and mx < 500 and my > 0 and my < 500: #check if mouse is within canvas
                if pickingColor: #set the color to that of the pixel if using colorPicker tool
                    color = canvas[int(mx / sizeOfPixel)][int(my / sizeOfPixel)].copy()
                    pickingColor = False
                else: #otherwise draw the pixel
                    canvas[int(mx / sizeOfPixel)][int(my / sizeOfPixel)] = color.copy()
                    mouseDrawing = True


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

    if pickingColor:
        window.blit(colorPickerCursor, (mx, my))  # draws colorPickerCursor icon
    elif usingEraser:
        window.blit(eraserCursor, (mx, my))  # draws eraserCursor icon
    else:
        window.blit(cursor, (mx, my))  # draws regular mouse icon


    # Update display
    manager.update(UI_REFRESH_RATE)
    manager.draw_ui(window)  # updates gui components (text input field)
    pygame.display.update()  #update all visuals

pygame.display.quit()
