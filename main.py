import clingo

from yolov2 import get_yolo_output

print(get_yolo_output())

  # Import the function from yolo.py

def get_neural_network_output():
    # Get YOLO detection results
    yolo_output = get_yolo_output()  # Calls yolo.py function
    return yolo_output



def run_asp_solver(light_color, cars_detected, turn_signal_on):
    # ASP program with placeholders for facts
    asp_program = f"""
    go :- light(green), not carsIncoming.
    go :- light(green), turnSignalOn.
    go :- light(yellow), not carsIncoming, turnSignalOn.
    :- light(red), go.
    :- light(yellow), carsIncoming, go.
    :- light(yellow), not turnSignalOn, go.

    light({light_color}).
    {'carsIncoming.' if cars_detected else '-carsIncoming.'}
    {'turnSignalOn.' if turn_signal_on else '-turnSignalOn.'}

    carsIncoming :- not -carsIncoming.
    turnSignalOn :- not -turnSignalOn.
    """

    # Create a Clingo control object
    ctl = clingo.Control()

    # Add the ASP program to the control object
    ctl.add("base", [], asp_program)

    # Ground the program (preparing it for solving)
    ctl.ground([("base", [])])

    # Solve and determine if it's safe to go
    result = ""
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            result = "safe" if "go" in str(model.symbols(shown=True)) else "not safe"

    return result


# Example live loop for real-time analysis
while True:
    # Get neural network output (replace with actual inference)
    nn_output = get_neural_network_output()

    # Run the ASP solver with neural network output
    decision = run_asp_solver(nn_output['light_color'], nn_output['cars_detected'], nn_output['turn_signal_on'])

    # Output the decision
    print(f"It is {decision} to take a left turn.")

    # Simulate a short delay (in a real system, this would run as frequently as needed)
    import time

    time.sleep(1)  # Adjust this based on how often the live feed updates
