/*
 * Auteur original du code: Marc De Falco
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <raylib.h>
#include <stdio.h>

#include "sequential.h"

double sigmoid(void *userdata, double x)
{ return 1.0 / (1 + exp(-x)); }

double sigmoid_derivate(void *userdata, double x, double y)
{ return y * (1. - y); }

Sequential_Actfun activation_function_finder(const char* name)
{
	Sequential_Actfun output;

	output.fun = NULL;
	output.derivate = NULL;

	if(!strcmp(name, "sigmoid")) {
		output.fun = sigmoid;
		output.derivate = sigmoid_derivate;
	}

	return output;
}

int main(int argc, char* argv[]) 
{
	if(argc != 2) {
		fprintf(stderr, "./mnist_demo [CHEMIN VERS UN MODELE]\n");
		return -1;
	}

	FILE *f = fopen(argv[1], "r");
	if(!f) {
		perror("fopen");
		return -1;
	}

	Sequential *model = sequential_read(f, activation_function_finder);
	fclose(f);

	Tensor scrible;
	if(tensor_new(&scrible, 28, 28, 1))
		return -1;
	
	tensor_clear(scrible);

    const int screenWidth = 800;
    const int screenHeight = 800;

    InitWindow(screenWidth, screenHeight, "DNN");

    Vector2 ballPosition = { -100.0f, -100.0f };
    Color ballColor = DARKBLUE;

    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
    //---------------------------------------------------------------------------------------

    while (!WindowShouldClose()) {  // Detect window close button or ESC key
        ballPosition = GetMousePosition();

        int x = ((int)ballPosition.x - 28 * 4) / 20;
        int y = ((int)ballPosition.y) / 20;

        if (0 <= x && x < 28 && 0 <= y && y < 28) {
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                for(int dx = 0; dx < 2; dx++) {
                    for(int dy = 0; dy < 2; dy++) {
                        if (0 <= x+dx && x+dx < 28 && 0 <= y+dy && y+dy < 28)
                            scrible.data[(y+dy) * scrible.width + x + dx] = 1;
					}
				}
			}
            if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) 
                scrible.data[y * scrible.width + x] = 0.;
        }

        if (IsKeyPressed(KEY_C))
			tensor_clear(scrible);

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();
		ClearBackground(RAYWHITE);

		for(x = 0; x < scrible.width; x++) {
			for(y = 0; y < scrible.height; y++) {
				char v = 255 * (1. - scrible.data[y * scrible.width + x]);
				Color c = (Color){ v, v, v, 255 };
				DrawRectangle(28 * 4 + 20*x, 20*y, 
					20, 20, c);
			}
		}

		DrawCircleV(ballPosition, 5, ballColor);

		char buf[1000];
		Tensor out = sequential_run(model, scrible);
		
		// Application de SoftMax
		double tmp = 0;
		for(int i = 0; i < out.width; i ++) {
			out.data[i] = exp(out.data[i]);
			tmp += out.data[i];
		}
		for(int i = 0; i < out.width; i ++)
			out.data[i] /= tmp;
		
		int max_index = 0;
		for(int i = 1; i < out.width; i ++) {
			if(out.data[i] > out.data[max_index])
				max_index = i;
		}

		for(int i = 0; i < out.width; i ++) {
			sprintf(buf, "%d(%lf %%)", i, 100. * out.data[i]);
			DrawText(buf, 10 + 150*(i % 5), 28 * 20 + 30 + 20*(i/5), 20,
				max_index == i ? GREEN : DARKGRAY);
		}
		
		DrawText("Press C to clear the screen",
			10, 28 * 20 + 90, 20,
			DARKGRAY
		);
		DrawText("Press the left mouse button to draw",
			10, 28 * 20 + 110, 20,
			DARKGRAY
		);
		DrawText("Press the right mouse button to erase",
			10, 28 * 20 + 130, 20,
			DARKGRAY
		);
        EndDrawing();
    }

    CloseWindow();
	tensor_free(scrible);
	sequential_free(model);
}
