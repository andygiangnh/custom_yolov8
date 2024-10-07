import subprocess

def convert_and_save_image(input_topic, output_topic, filename):
    # Start the image_proc node
    proc = subprocess.Popen([
        'rosrun', 'image_proc', 'image_proc',
        f'image:={input_topic}', '_image_transport:=raw'
    ])
    
    # Save the converted image
    command = [
        'rosrun', 'image_view', 'image_saver',
        f'image:={output_topic}', '-n', '1',
        f'_filename_format:={filename}_%04d.jpg'
    ]
    subprocess.run(command)
    
    # Terminate the image_proc node
    proc.terminate()

if __name__ == "__main__":
    input_image_topic = "/camera/depth/image_raw"  # Replace with your input image topic
    output_image_topic = "/camera/depth/image_rgb"  # Replace with your output image topic
    image_name = "my_image"  # Replace with your desired image name
    convert_and_save_image(input_image_topic, output_image_topic, image_name)
