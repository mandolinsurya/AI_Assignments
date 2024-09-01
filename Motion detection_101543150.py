import cv2
import numpy as np
import os

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to read camera feed")

else:
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # Capture the first frame
    ret, frame = cap.read()
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Save the grayscale image as the reference background
        cv2.imwrite('reference_background.jpg', gray)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Compute the absolute difference with the reference background
            diff_frame = cv2.absdiff(gray, gray_frame)
            # Apply a threshold
            _, thresholded_diff = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
            # Find contours
            contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Check if contour area is large
                if cv2.contourArea(contour) > 10000:
                    # Draw rectangle around the object
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Output "UNSAFE" text in red color
                    cv2.putText(frame, "UNSAFE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    # Save the unsafe image
                    cv2.imwrite('unsafe_image.jpg', frame)
                    # Give out an audio alert
                    os.system('say "Alert! Unsafe condition detected."')
                    # Generate an alarm sound
                    os.system('say "beep"')
            # Write the frame into the file 'output.avi'
            out.write(frame)
            # Display the resulting frame
            cv2.imshow('Webcam Live Video', frame)
            # Display the thresholded difference frame
            cv2.imshow('Thresholded Difference Frame', thresholded_diff)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    # Delete the created images
    os.remove('reference_background.jpg')
    os.remove('unsafe_image.jpg')
