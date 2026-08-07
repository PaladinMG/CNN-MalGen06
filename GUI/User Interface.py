from __future__ import annotations
import tkinter as tk
import customtkinter as ct
import numpy as np
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
import napari as nap
import argparse
from typing import Literal

from collections.abc import Callable
from functools import partial

from aicspylibczi import CziFile
from PIL import Image
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)


# def button_callback():
#     print("Button Pressed")

# def open_ui():
#     app = ct.CTk()
#     app.title('Setup')
#     app.resizable(width=True, height=True)
#     app.geometry('400x400')
#     button = ct.CTkButton(app, text='Select Main Directory \n (Must be empty)', command=button_callback)
#     button.grid(row=0, column=0, padx=20, pady=20, sticky='ew', columnspan=2)



# def main():
#     app = ct.CTk()
#     app.mainloop()

# if __name__ == "__main__":
#     main()

class FolderSelectFrame(ct.CTkFrame):
    """
    Frame containing file selection dialog
    """

    def __init__(self, master, text: str):
        super().__init__(master)

        self.app: App = master
        self.text: str = text
        self.filename: str = ""
        self.dir: Path | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.closed_folder_icon = tk.PhotoImage(
            file="icons/folder-closed-solid.png"
        )
        self.open_folder_icon = tk.PhotoImage(
            file="icons/folder-open-solid.png"
        )

        self.button = ct.CTkButton(
            self,
            text=self.text,
            command=self.select_folder,
        )
        self.button.grid(
            row=0,
            column=0,
            padx=20,
            pady=20,
            sticky="ew",
            columnspan=2,
        )

        self.tree = ttk.Treeview(
            self,
            show="tree",
        )
        self.tree.grid(
            row=1,
            column=0,
            padx=20,
            pady=(0, 20),
            columnspan=2,
            sticky="nsew",
        )

        self.tree.bind(
            "<<TreeviewOpen>>",
            self.folder_opened,
        )
        self.tree.bind(
            "<<TreeviewClose>>",
            self.folder_closed,
        )

    def folder_opened(self, event):
        node = self.tree.focus()

        if node:
            self.tree.item(
                node,
                image=self.open_folder_icon,
            )

    def folder_closed(self, event):
        node = self.tree.focus()

        if node:
            self.tree.item(
                node,
                image=self.closed_folder_icon,
            )

    def populate_tree(self, root_path: Path | None):
        if root_path is None:
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        parent_node = ""

        for path_part in root_path.parts:
            parent_node = self.tree.insert(
                parent_node,
                "end",
                text=path_part,
                image=self.open_folder_icon,
                open=True,
            )

    def select_folder(self):
        folder_path = filedialog.askdirectory(
            title="Select Folder",
            parent=self,
        )

        if not folder_path:
            return

        selected_dir = Path(folder_path)

        # Raise an error if the selected directory contains anything.
        if any(selected_dir.iterdir()):
            raise ValueError(
                f"The selected folder is not empty: {selected_dir}"
            )

        self.dir = selected_dir
        self.populate_tree(self.dir)

        self.app.forward.configure(state='normal')

class CZISelectFrame(ct.CTkFrame):
    """
    Frame containing a folder-selection dialog for CZI files.
    """

    def __init__(self, master, text: str):
        super().__init__(master)

        self.app: App = master
        self.text: str = text
        self.filename: str = ""
        self.dir: Path | None = None
        self.czi_list: list[Path] = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.closed_folder_icon = tk.PhotoImage(
            file="icons/folder-closed-solid.png"
        )
        self.open_folder_icon = tk.PhotoImage(
            file="icons/folder-open-solid.png"
        )

        self.button = ct.CTkButton(
            self,
            text=self.text,
            command=self.select_folder,
        )
        self.button.grid(
            row=0,
            column=0,
            padx=20,
            pady=20,
            sticky="ew",
            columnspan=2,
        )

        self.tree = ttk.Treeview(
            self,
            show="tree",
        )
        self.tree.grid(
            row=1,
            column=0,
            padx=20,
            pady=(0, 20),
            columnspan=2,
            sticky="nsew",
        )

        self.tree.bind(
            "<<TreeviewOpen>>",
            self.folder_opened,
        )
        self.tree.bind(
            "<<TreeviewClose>>",
            self.folder_closed,
        )

    def folder_opened(self, event):
        node = self.tree.focus()

        if node:
            self.tree.item(
                node,
                image=self.open_folder_icon,
            )

    def folder_closed(self, event):
        node = self.tree.focus()

        if node:
            self.tree.item(
                node,
                image=self.closed_folder_icon,
            )

    def populate_tree(self, root_path: Path | None):
        if root_path is None:
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        root_node = self.tree.insert(
            "",
            "end",
            text=root_path.name or str(root_path),
            image=self.open_folder_icon,
            open=True,
        )

        for czi_file in self.czi_list:
            self.tree.insert(
                root_node,
                "end",
                text=czi_file.name,
                values=(str(czi_file),),
            )

    def select_folder(self):
        folder_path = filedialog.askdirectory(
            title="Select Folder",
            parent=self,
        )

        if not folder_path:
            return

        selected_dir = Path(folder_path)

        self.czi_list = sorted(
            (
                path
                for path in selected_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".czi"
            ),
            key=lambda path: path.name.lower(),
        )

        if not self.czi_list:
            raise ValueError(
                f"The selected folder contains no .czi files: "
                f"{selected_dir}"
            )

        self.dir = selected_dir
        self.populate_tree(self.dir)

        self.app.forward.configure(state="normal")

class GetPatchesFrame(ct.CTkFrame):
    """
    Frame used to select and manage training patches from CZI scans.

    The napari viewer displays each CZI at 1/8 resolution. Clicking the
    image places a square representing a 4096 x 4096 full-resolution patch.
    """

    PATCH_SIZE = 4096
    DISPLAY_SCALE = 1 / 8
    DISPLAY_PATCH_SIZE = PATCH_SIZE * DISPLAY_SCALE

    def __init__(
        self,
        master: "App",
        text: str,
        root_dir_getter: Callable[[], Path | None],
        czi_list_getter: Callable[[], list[Path]],
    ):
        super().__init__(master)

        self.app: App = master
        self.text = text

        # These are callables because the directories and CZI list may not
        # exist yet when this frame is first constructed.
        self.root_dir_getter = root_dir_getter
        self.czi_list_getter = czi_list_getter

        self.patch_count = 0
        self.current_czi_index = 0

        self.viewer: nap.Viewer | None = None
        self.image_layer = None
        self.rectangle_layer = None
        self.napari_controls = None

        self.current_overview: np.ndarray | None = None
        self.current_czi_path: Path | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.select_button = ct.CTkButton(
            self,
            text="Select Training Patches",
            command=self.open_patch_selector,
        )
        self.select_button.grid(
            row=0,
            column=0,
            padx=(20, 10),
            pady=(20, 10),
            sticky="ew",
        )

        self.edit_button = ct.CTkButton(
            self,
            text="Edit",
            command=self.open_edit_dialog,
        )
        self.edit_button.grid(
            row=0,
            column=1,
            padx=(0, 20),
            pady=(20, 10),
            sticky="ew",
        )

        self.patch_textbox = ct.CTkTextbox(
            self,
            height=70,
        )
        self.patch_textbox.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=20,
            pady=(0, 20),
            sticky="nsew",
        )

        self.refresh_patch_count()

    def get_images_directory(self) -> Path:
        """
        Return root/dataset/images, creating it when necessary.
        """
        root_dir = self.root_dir_getter()

        if root_dir is None:
            raise ValueError(
                "A main output directory must be selected first."
            )

        images_dir = root_dir / "dataset" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        return images_dir

    def get_czi_list(self) -> list[Path]:
        czi_list = self.czi_list_getter()

        if not czi_list:
            raise ValueError("No CZI files have been selected.")

        return czi_list

    def refresh_patch_count(self):
        """
        Recount exported PNG patches and update the interface.

        The application's forward button is enabled only when at least
        two exported patches exist.
        """
        try:
            images_dir = self.get_images_directory()
        except ValueError:
            self.patch_count = 0
        else:
            self.patch_count = sum(
                1
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            )

        self.patch_textbox.configure(state="normal")
        self.patch_textbox.delete("1.0", "end")
        self.patch_textbox.insert(
            "1.0",
            f"Current Patches: {self.patch_count}",
        )
        self.patch_textbox.configure(state="disabled")

        self.app.forward.configure(
            state="normal" if self.patch_count > 1 else "disabled"
        )

    def open_patch_selector(self):
        czi_list = self.get_czi_list()
        self.get_images_directory()

        if self.current_czi_index >= len(czi_list):
            self.current_czi_index = 0

        self.viewer = nap.Viewer(
            title="Training Patch Selection"
        )

        self.rectangle_layer = self.viewer.add_shapes(
            name="Selected patches",
            shape_type="rectangle",
            edge_width=3,
            edge_color="red",
            face_color="transparent",
        )

        self.viewer.mouse_drag_callbacks.append(
            self.place_patch
        )

        self.napari_controls = PatchSelectionControls(self)

        self.viewer.window.add_dock_widget(
            self.napari_controls,
            name="Patch Selection",
            area="right",
        )

        self.load_current_czi()

        # This blocks the Tkinter window while napari is open.
        # Execution resumes after the napari viewer closes.
        nap.run()

        self.viewer = None
        self.image_layer = None
        self.rectangle_layer = None
        self.napari_controls = None

        self.refresh_patch_count()

    def read_czi_overview(self, czi_path: Path) -> np.ndarray:
        czi = CziFile(czi_path)

        if not czi.is_mosaic():
            raise ValueError(
                f"This implementation expects a mosaic CZI: {czi_path}"
            )

        overview = czi.read_mosaic(
            C=0,
            scale_factor=self.DISPLAY_SCALE,
        )

        overview = np.squeeze(overview)

        return self.prepare_rgb_array(overview)

    @staticmethod
    def prepare_rgb_array(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        image = np.squeeze(image)

        if image.ndim == 2:
            return image

        if image.ndim != 3:
            raise ValueError(
                "Expected a 2D grayscale image or 3D color image, "
                f"but received shape {image.shape}."
            )

        # Convert C, Y, X to Y, X, C.
        if image.shape[0] in (3, 4):
            image = np.moveaxis(image, 0, -1)

        # Remove alpha if present.
        if image.shape[-1] == 4:
            image = image[..., :3]

        if image.shape[-1] != 3:
            raise ValueError(
                "Could not determine the color-sample dimension for "
                f"shape {image.shape}."
            )

        # CZI BGR → RGB
        image = image[..., ::-1].copy()

        return image

    def load_current_czi(self):
        if self.viewer is None:
            return

        czi_list = self.get_czi_list()
        self.current_czi_path = czi_list[self.current_czi_index]

        self.current_overview = self.read_czi_overview(
            self.current_czi_path
        )

        if self.image_layer is None:
            self.image_layer = self.viewer.add_image(
                self.current_overview,
                name=self.current_czi_path.name,
                rgb=(
                    self.current_overview.ndim == 3
                    and self.current_overview.shape[-1] == 3
                ),
            )
        else:
            self.image_layer.data = self.current_overview
            self.image_layer.name = self.current_czi_path.name

        self.clear_rectangles()

        self.viewer.reset_view()

        if self.napari_controls is not None:
            self.napari_controls.update_scan_label()
            self.napari_controls.update_navigation_buttons()
            self.napari_controls.update_selection_label()

    def place_patch(self, viewer, event):
        """
        Place one fixed-size rectangle centered at the mouse pointer.
        """
        if event.type != "mouse_press":
            return

        # Restrict placement to the primary mouse button.
        if event.button != 1:
            return

        if self.image_layer is None:
            return

        if self.rectangle_layer is None:
            return

        # event.position is expressed in world coordinates.
        data_position = self.image_layer.world_to_data(
            event.position
        )

        center_y = float(data_position[-2])
        center_x = float(data_position[-1])

        half_size = self.DISPLAY_PATCH_SIZE / 2

        y_min = center_y - half_size
        y_max = center_y + half_size
        x_min = center_x - half_size
        x_max = center_x + half_size

        image_height = self.image_layer.data.shape[0]
        image_width = self.image_layer.data.shape[1]

        # Reject patches that would extend beyond the CZI.
        if (
            y_min < 0
            or x_min < 0
            or y_max > image_height
            or x_max > image_width
        ):
            if self.napari_controls is not None:
                self.napari_controls.set_status(
                    "That patch would extend beyond the scan boundary."
                )
            return

        rectangle = np.array(
            [
                [y_min, x_min],
                [y_min, x_max],
                [y_max, x_max],
                [y_max, x_min],
            ],
            dtype=float,
        )

        self.rectangle_layer.add_rectangles(
            rectangle,
            edge_color="red",
            face_color="transparent",
        )

        if self.napari_controls is not None:
            self.napari_controls.set_status(
                "Patch added."
            )
            self.napari_controls.update_selection_label()

    def clear_rectangles(self):
        if self.rectangle_layer is None:
            return

        self.rectangle_layer.data = []

        if self.napari_controls is not None:
            self.napari_controls.update_selection_label()

    def read_full_resolution_patch(
            self,
            czi_path: Path,
            x: int,
            y: int,
    ) -> np.ndarray:
        czi = CziFile(czi_path)

        patch = czi.read_mosaic(
            C=0,
            region=(
                x,
                y,
                self.PATCH_SIZE,
                self.PATCH_SIZE,
            ),
            scale_factor=1.0,
        )

        patch = np.squeeze(patch)

        return self.prepare_rgb_array(patch)

    @staticmethod
    def convert_to_uint8(image: np.ndarray) -> np.ndarray:
        """
        Convert an image array to uint8 for PNG export.
        """
        image = np.asarray(image)

        if image.dtype == np.uint8:
            return image

        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max

            if max_value == 0:
                return np.zeros_like(image, dtype=np.uint8)

            image = image.astype(np.float32) / max_value
            image = image * 255

            return np.clip(image, 0, 255).astype(np.uint8)

        image = image.astype(np.float32)

        finite_values = image[np.isfinite(image)]

        if finite_values.size == 0:
            return np.zeros(image.shape, dtype=np.uint8)

        low, high = np.percentile(
            finite_values,
            (1, 99),
        )

        if high <= low:
            return np.zeros(image.shape, dtype=np.uint8)

        image = (image - low) / (high - low)
        image = image * 255

        return np.clip(image, 0, 255).astype(np.uint8)

    def export_current_patches(self):
        if self.rectangle_layer is None:
            return

        if self.current_czi_path is None:
            raise ValueError("No CZI scan is currently loaded.")

        rectangles = list(self.rectangle_layer.data)

        if not rectangles:
            if self.napari_controls is not None:
                self.napari_controls.set_status(
                    "No patches have been selected."
                )
            return

        images_dir = self.get_images_directory()

        existing_numbers = []

        for path in images_dir.glob("patch_*.png"):
            try:
                existing_numbers.append(
                    int(path.stem.rsplit("_", 1)[-1])
                )
            except ValueError:
                continue

        next_number = (
            max(existing_numbers, default=0) + 1
        )

        exported_count = 0

        for rectangle in rectangles:
            rectangle = np.asarray(rectangle)

            overview_y_min = float(rectangle[:, 0].min())
            overview_x_min = float(rectangle[:, 1].min())

            full_y = round(
                overview_y_min / self.DISPLAY_SCALE
            )
            full_x = round(
                overview_x_min / self.DISPLAY_SCALE
            )

            patch = self.read_full_resolution_patch(
                czi_path=self.current_czi_path,
                x=full_x,
                y=full_y,
            )

            patch = self.convert_to_uint8(patch)

            output_name = (
                f"{self.current_czi_path.stem}"
                f"_patch_{next_number:05d}.png"
            )

            output_path = images_dir / output_name

            Image.fromarray(patch).save(output_path)

            next_number += 1
            exported_count += 1

        self.clear_rectangles()
        self.refresh_patch_count()

        if self.napari_controls is not None:
            self.napari_controls.set_status(
                f"Exported {exported_count} patch"
                f"{'' if exported_count == 1 else 'es'}."
            )
    def previous_czi(self):
        if self.current_czi_index <= 0:
            return

        self.current_czi_index -= 1
        self.load_current_czi()

    def next_czi(self):
        czi_list = self.get_czi_list()

        if self.current_czi_index >= len(czi_list) - 1:
            return

        self.current_czi_index += 1
        self.load_current_czi()

    def close_napari(self):
        if self.viewer is not None:
            self.viewer.close()

    def open_edit_dialog(self):
        try:
            images_dir = self.get_images_directory()
        except ValueError as error:
            messagebox.showerror(
                title="No Output Directory",
                message=str(error),
                parent=self,
            )
            return

        png_files = sorted(
            images_dir.glob("*.png"),
            key=lambda path: path.name.lower(),
        )

        dialog = ct.CTkToplevel(self)
        dialog.title("Edit Exported Patches")
        dialog.geometry("600x500")
        dialog.transient(self.app)
        dialog.grab_set()

        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(1, weight=1)

        title = ct.CTkLabel(
            dialog,
            text="Select exported patches to remove:",
        )
        title.grid(
            row=0,
            column=0,
            padx=20,
            pady=(20, 10),
            sticky="w",
        )

        scroll_frame = ct.CTkScrollableFrame(dialog)
        scroll_frame.grid(
            row=1,
            column=0,
            padx=20,
            pady=10,
            sticky="nsew",
        )
        scroll_frame.grid_columnconfigure(0, weight=1)

        selected_files: dict[Path, tk.BooleanVar] = {}

        if not png_files:
            no_files_label = ct.CTkLabel(
                scroll_frame,
                text="No exported PNG patches were found.",
            )
            no_files_label.grid(
                row=0,
                column=0,
                padx=10,
                pady=10,
                sticky="w",
            )

        for row, path in enumerate(png_files):
            selected = tk.BooleanVar(value=False)
            selected_files[path] = selected

            checkbox = ct.CTkCheckBox(
                scroll_frame,
                text=path.name,
                variable=selected,
            )
            checkbox.grid(
                row=row,
                column=0,
                padx=10,
                pady=5,
                sticky="w",
            )

        button_frame = ct.CTkFrame(
            dialog,
            fg_color="transparent",
        )
        button_frame.grid(
            row=2,
            column=0,
            padx=20,
            pady=(10, 20),
            sticky="ew",
        )
        button_frame.grid_columnconfigure((0, 1), weight=1)

        delete_button = ct.CTkButton(
            button_frame,
            text="Delete Selected",
            command=partial(
                self.delete_selected_patches,
                selected_files,
                dialog,
            ),
        )
        delete_button.grid(
            row=0,
            column=0,
            padx=(0, 5),
            sticky="ew",
        )

        cancel_button = ct.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
        )
        cancel_button.grid(
            row=0,
            column=1,
            padx=(5, 0),
            sticky="ew",
        )

    def delete_selected_patches(
        self,
        selected_files: dict[Path, tk.BooleanVar],
        dialog: ct.CTkToplevel,
    ):
        files_to_delete = [
            path
            for path, selected in selected_files.items()
            if selected.get()
        ]

        if not files_to_delete:
            return

        confirmed = messagebox.askyesno(
            title="Delete Patches",
            message=(
                f"Delete {len(files_to_delete)} selected patch"
                f"{'' if len(files_to_delete) == 1 else 'es'}?"
            ),
            parent=dialog,
        )

        if not confirmed:
            return

        for path in files_to_delete:
            try:
                path.unlink()
            except FileNotFoundError:
                continue

        dialog.destroy()
        self.refresh_patch_count()

class PatchSelectionControls(QWidget):
    """
    Dock widget containing the instructions and controls for napari.
    """

    def __init__(self, patch_frame: GetPatchesFrame):
        super().__init__()

        self.patch_frame = patch_frame

        layout = QVBoxLayout()
        self.setLayout(layout)

        instructions = QLabel(
            "Instructions:\n\n"
            "1. Left-click the scan to place a centered "
            "4096 × 4096 patch.\n"
            "2. Place as many patches as needed.\n"
            "3. Click Export Current Patches to save all "
            "rectangles from this scan.\n"
            "4. Use Previous Scan and Next Scan to move "
            "between CZI files.\n"
            "5. Click Done when patch selection is complete."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self.scan_label = QLabel()
        self.scan_label.setWordWrap(True)
        layout.addWidget(self.scan_label)

        self.selection_label = QLabel()
        layout.addWidget(self.selection_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        export_button = QPushButton(
            "Export Current Patches"
        )
        export_button.clicked.connect(
            self.patch_frame.export_current_patches
        )
        layout.addWidget(export_button)

        clear_button = QPushButton(
            "Clear Unexported Rectangles"
        )
        clear_button.clicked.connect(
            self.patch_frame.clear_rectangles
        )
        layout.addWidget(clear_button)

        navigation_layout = QHBoxLayout()

        self.previous_button = QPushButton(
            "Previous Scan"
        )
        self.previous_button.clicked.connect(
            self.patch_frame.previous_czi
        )
        navigation_layout.addWidget(
            self.previous_button
        )

        self.next_button = QPushButton(
            "Next Scan"
        )
        self.next_button.clicked.connect(
            self.patch_frame.next_czi
        )
        navigation_layout.addWidget(
            self.next_button
        )

        layout.addLayout(navigation_layout)

        done_button = QPushButton("Done")
        done_button.clicked.connect(
            self.patch_frame.close_napari
        )
        layout.addWidget(done_button)

        self.update_scan_label()
        self.update_navigation_buttons()
        self.update_selection_label()

    def update_scan_label(self):
        czi_list = self.patch_frame.get_czi_list()
        index = self.patch_frame.current_czi_index
        czi_path = czi_list[index]

        self.scan_label.setText(
            f"Current scan: {index + 1} of {len(czi_list)}\n"
            f"{czi_path.name}"
        )

    def update_navigation_buttons(self):
        czi_list = self.patch_frame.get_czi_list()
        index = self.patch_frame.current_czi_index

        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(
            index < len(czi_list) - 1
        )

    def update_selection_label(self):
        layer = self.patch_frame.rectangle_layer

        selection_count = (
            len(layer.data)
            if layer is not None
            else 0
        )

        self.selection_label.setText(
            f"Unexported selections: {selection_count}"
        )

    def set_status(self, message: str):
        self.status_label.setText(message)


class PlaceholderFrame(ct.CTkFrame):
    def __init__(self, master):
        super().__init__(None)

class Frames:
    def __init__(self, master: "App"):
        self.folder_select_frame = FolderSelectFrame(
            master,
            "Select Main Directory\n(must be empty)",
        )

        self.czi_select_frame = CZISelectFrame(
            master,
            "Select CZI Directory",
        )

        self.get_patches_frame = GetPatchesFrame(
            master=master,
            text="Select Training Patches",
            root_dir_getter=lambda: self.folder_select_frame.dir,
            czi_list_getter=lambda: self.czi_select_frame.czi_list,
        )

        self.frame_list: list[ct.CTkFrame] = [
            self.folder_select_frame,
            self.czi_select_frame,
            self.get_patches_frame,
        ]


class App(ct.CTk):
    def __init__(self):
        super().__init__()

        self.forward = ct.CTkButton(self, text='->', command=lambda: self.initialize_frame('forward'), state="disabled")
        self.backward = ct.CTkButton(self, text='<-', command=lambda: self.initialize_frame('backward'))
        self.forward.grid(row=1, column=1, padx=(5, 10), pady=(0,20), sticky="ew")
        self.backward.grid(row=1, column=0, padx=(10, 5), pady=(0, 20), sticky="ew")

        self.frame_list_: list[ct.CTkFrame] = Frames(self).frame_list
        self.current_frame: tuple[int, ct.CTkFrame] = (-1, PlaceholderFrame(self))

        self.initialize_frame(direction="forward")

    def initialize_frame(
            self,
            direction: Literal["forward", "backward"],
    ):
        if direction not in ("forward", "backward"):
            raise ValueError(f"Invalid direction: {direction}")

        step = 1 if direction == "forward" else -1
        new_index = self.current_frame[0] + step

        if not 0 <= new_index < len(self.frame_list_):
            return

        self.current_frame[1].grid_remove()

        self.current_frame = (
            new_index,
            self.frame_list_[new_index],
        )

        self.current_frame[1].grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="nsew",
        )

        self.backward.configure(
            state="disabled" if new_index == 0 else "normal"
        )

        if new_index == len(self.frame_list_) - 1:
            self.forward.configure(state="disabled")
        elif isinstance(self.current_frame[1], FolderSelectFrame):
            self.forward.configure(
                state="normal"
                if self.current_frame[1].dir is not None
                else "disabled"
            )
        else:
            self.forward.configure(state="normal")






def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
