import pygame
import pygame_gui

def between(a,b,c): #return if a is between b and c
    return a >= b and a <= c

def eraser():
    color[0] = 200
    color[1] = 200
    color[2] = 200

def setColor():
    if redInput.get_text().isnumeric() and greenInput.get_text().isnumeric() and blueInput.get_text().isnumeric() and between(int(redInput.get_text()), 0, 255) and between(int(greenInput.get_text()), 0, 255) and between(int(blueInput.get_text()), 0, 255):
            color[0] = int(redInput.get_text())
            color[1] = int(greenInput.get_text())
            color[2] = int(blueInput.get_text())

            redInput.clear()
            greenInput.clear()
            blueInput.clear()

def confirm(): #confirms the current drawing
    print("Confirm function called")

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


color = [0,0,0]
windowSideLength = 500 #window is a square so this is both length and width
sidePanelWidth = 150
numberOfPixels = 20 #the number of pixels in each dimension (ex. 28 means the screen is 28x28 pixels)
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

        if event.type == pygame.MOUSEBUTTONDOWN:
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