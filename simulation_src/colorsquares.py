import random
from PIL import Image, ImageDraw
import argparse
import torch
import torchvision.transforms as transforms

# Define the colors
colors = ["white", "blue", "green", "yellow", "red", "pink", "orange", "purple", "cyan", "magenta", "brown"] # for color change detect
#colors = ['green'] * 10 # for location change detect

# Define the image size and square size
image_size = (500, 500)
square_size = 60
buffer = 5

def generate_color_squares_image(num_squares):
    # Create a black background image
    image = Image.new("RGB", image_size, "black")
    draw = ImageDraw.Draw(image)

    frames = []
    for i in range(1,num_squares+1):
        frames += [Image.new("RGB", image_size, "black")]

    
    # Get random positions for the squares
    positions = []
    while len(positions) < num_squares:
        x = random.randint(0, image_size[0] - square_size)
        y = random.randint(0, image_size[1] - square_size)
        if not any(abs(x - px) < square_size+buffer and abs(y - py) < square_size+buffer for px, py in positions):
            positions.append((x, y))
    #print(positions)
    # Draw the squares with random colors
    i = 0
    selected_colors = random.sample(colors, num_squares)
    for position, color in zip(positions, selected_colors):
        cur_frame = frames[i]
        draw_frame = ImageDraw.Draw(cur_frame)

        draw.rectangle([position, (position[0] + square_size, position[1] + square_size)], fill=color)
        draw_frame.rectangle([position, (position[0] + square_size, position[1] + square_size)], fill=color)
        frames[i] = cur_frame.resize((28,28))
        i += 1
    
    return image, positions, selected_colors, frames

def generate_change_image(original_positions, original_colors):
    # Create a new black background image
    image = Image.new("RGB", image_size, "black")
    draw = ImageDraw.Draw(image)

    frames = []
    for i in range(1,len(original_positions)+1):
        frames += [Image.new("RGB", image_size, "black")]
    
    # Choose a square to change its color
    change_index = 0#random.randint(0, 4) #len(original_colors) - 1
    new_color_choices = [color for color in colors if color not in original_colors]
    new_color = random.choice(new_color_choices)
    
    # Draw the squares with one color changed
    for i, (position, color) in enumerate(zip(original_positions, original_colors)):
        cur_frame = frames[i]
        draw_frame = ImageDraw.Draw(cur_frame)
        if i == change_index:
            # generate valid new image
            new_position = position #original_positions[change_index] # position
            #while any(abs(new_position[0] - px) < square_size+buffer and abs(new_position[1] - py) < square_size+buffer for px, py in original_positions):
             #   new_position = (random.randint(0, image_size[0] - square_size), random.randint(0, image_size[1] - square_size))
            
            draw.rectangle([new_position, (new_position[0] + square_size, new_position[1] + square_size)], fill=new_color) #new_color V
            draw_frame.rectangle([new_position, (new_position[0] + square_size, new_position[1] + square_size)], fill=color)
            frames[i] = cur_frame.resize((28,28))
        else:
            draw.rectangle([position, (position[0] + square_size, position[1] + square_size)], fill=color)
            draw_frame.rectangle([position, (position[0] + square_size, position[1] + square_size)], fill=color)
            frames[i] = cur_frame.resize((28,28))
    
    return image, frames

def save_images(original_image, change_image, index):
    original_image.save(f"original_image_{index}.png")
    change_image.save(f"change_image_{index}.png")

def main(args):
    num_images, num_squares, to_tensor, change =args.num_images, args.num_squares, args.to_tensor, args.change
#    num_images = 5  # Number of images to generate
#    num_squares = random.randint(1, 8)
    if to_tensor is False:
        for i in range(num_images%11):
            original_image, positions, colors, original_frames = generate_color_squares_image(num_squares)
            change_image, change_frames = generate_change_image(positions, colors)
            save_images(original_image, change_image, i)
    else:
        for set_size in range(1, num_squares+1):
            totensor = transforms.ToTensor()
            original_image, positions, colors, original_frames = generate_color_squares_image(set_size)
            change_image, change_frames = generate_change_image(positions, colors)
            original_image, change_image = original_image.resize((28,28)), change_image.resize((28,28))
            original, change =  totensor(original_image).view(1,3,28,28), totensor(change_image).view(1,3,28,28)
            original_frames_out = torch.stack([totensor(frame) for frame in original_frames]).view(1,set_size,3,28,28)
            change_frames_out = torch.stack([totensor(frame) for frame in change_frames]).view(1,set_size,3,28,28)
            out_positions = [positions]
            print('frames',original_frames_out.size())

            for i in range(1, num_images):
                original_image, positions, colors, original_frames = generate_color_squares_image(set_size)
                change_image, change_frames = generate_change_image(positions, colors)
                original_image, change_image = original_image.resize((28,28)), change_image.resize((28,28))
                original_image, change_image =  totensor(original_image).view(1,3,28,28), totensor(change_image).view(1,3,28,28)
                original_frames = torch.stack([totensor(frame) for frame in original_frames]).view(1,set_size,3,28,28)
                change_frames = torch.stack([totensor(frame) for frame in change_frames]).view(1,set_size,3,28,28)

                original = torch.cat([original, original_image],dim=0)
                change = torch.cat([change, change_image],dim=0)
                original_frames_out = torch.cat([original_frames_out, original_frames],dim=0)
                change_frames_out = torch.cat([change_frames_out, change_frames],dim=0)
                out_positions += [positions]

            
            torch.save(original, f'original_{set_size}_color.pth')
            torch.save(change, f'change_{set_size}_color.pth')
            torch.save(original_frames_out, f'original_frames_{set_size}_color.pth')
            torch.save(change_frames_out, f'change_frames_{set_size}_color.pth')
            torch.save(out_positions, f'positions_{set_size}_color.pth')
            print(original.size())




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate color square images for change detection paradigms.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to generate")
    parser.add_argument("--num_squares", type=int, default=4, help="Number of squares in each image")
    parser.add_argument("--to_tensor", type=bool, default=False, help="save as tensor")
    parser.add_argument("--change", type=str, default='color', help="change type")
    args = parser.parse_args()
    
    main(args)