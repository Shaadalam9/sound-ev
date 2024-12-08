using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class menuscript : MonoBehaviour
{
    public GameObject screen1; // Reference to Screen_1 Canvas
    public GameObject screen;  // Reference to Screen Canvas
    private bool isFirstClick = true; // Track if it's the first click

    public void StartBtn()
    {
        if (isFirstClick)
        {
            // Show Screen_1 and hide the main Screen
            screen.SetActive(false);
            screen1.SetActive(true);
            isFirstClick = false;
        }
        else
        {
            // Load the environment scene on the second click
            SceneManager.LoadScene("Environment");
        }
    }

    public void ExitBtn()
    {
        Application.Quit();
    }
}