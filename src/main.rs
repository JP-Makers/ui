#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

slint::include_modules!();

use std::thread;
use rfd::FileDialog;

fn main() {
    let main_window = MainWindow::new().unwrap();

    let weak_win1 = main_window.as_weak();
    main_window.on_open_file1(move || {
        let weak = weak_win1.clone();
        thread::spawn(move || {
            if let Some(path) = FileDialog::new().set_title("Open File 1").pick_file() {
                let path_str = path.display().to_string();
                if let Some(win) = weak.upgrade() {
                    win.set_file1_path(path_str.into());
                }
            }
        });
    });

    let weak_win2 = main_window.as_weak();
    main_window.on_open_file2(move || {
        let weak = weak_win2.clone();
        thread::spawn(move || {
            if let Some(path) = FileDialog::new().set_title("Open File 2").pick_file() {
                let path_str = path.display().to_string();
                if let Some(win) = weak.upgrade() {
                    win.set_file2_path(path_str.into());
                }
            }
        });
    });

    main_window.run().unwrap();
}
