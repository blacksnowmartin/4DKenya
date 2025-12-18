import turtle
import colorsys

# Set up the screen
screen = turtle.Screen()
screen.bgcolor("black")
# Use the full RGB range from 0.0 to 1.0 for colorsys
turtle.colormode(1.0) 

# Create the turtle object
t = turtle.Turtle()
t.speed(0) # Set speed to the fastest
t.width(2)
t.hideturtle() # Hide the turtle cursor for a cleaner look

# Loop to draw the pattern
for i in range(360):
    # Generate colors based on the current loop iteration (hue transition)
    hue = i / 360.0
    color = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    t.pencolor(color)

    # Draw a line and turn
    t.forward(i)
    t.right(90.991) # A specific angle that creates an intricate, repeating pattern

# Keep the window open until clicked
screen.exitonclick()
