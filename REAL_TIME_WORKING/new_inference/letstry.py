from depthai import main
import cv2
if __name__ == "__main__":
    main()
    for frame in main():
        # Process the frame
        cv2.imshow("RGB Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()