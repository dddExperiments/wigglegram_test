import os
import sys

def process_includes(filepath, processed_files=None):
    if processed_files is None:
        processed_files = set()
    
    filepath = os.path.normpath(filepath)
    if filepath in processed_files:
        return ""
    processed_files.add(filepath)
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return f"// File not found: {filepath}\n"

    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_content = []
    shader_dir = os.path.dirname(filepath)
    
    for line in lines:
        if line.strip().startswith('#include "'):
            include_path = line.split('"')[1]
            full_include_path = os.path.normpath(os.path.join(shader_dir, include_path))
            new_content.append(process_includes(full_include_path, processed_files))
        else:
            new_content.append(line)
            
    return "".join(new_content)

def main(shader_dir, output_file):
    print(f"Embedding shaders from {shader_dir} into {output_file}")
    
    # Ensure shader_dir is absolute to avoid confusion
    shader_dir = os.path.abspath(shader_dir)
    
    with open(output_file, 'w') as f:
        f.write("#pragma once\n")
        f.write("#include <unordered_map>\n")
        f.write("#include <string>\n\n")
        f.write("namespace shader_embed {\n")
        f.write("    static const std::unordered_map<std::string, std::string> shaders = {\n")
        
        for root, dirs, files in os.walk(shader_dir):
            for filename in files:
                if filename.endswith(".wgsl"):
                    filepath = os.path.join(root, filename)
                    
                    # Compute relative path from shader_dir
                    rel_path = os.path.relpath(filepath, shader_dir).replace("\\", "/")
                    
                    # Remap keys for C++ compatibility
                    # Old: default/blur.wgsl, Now: detection/default/blur.wgsl
                    if rel_path.startswith("detection/"):
                        rel_path = rel_path[len("detection/"):]
                    # Old: matcher.wgsl, Now: matching/matcher.wgsl
                    if rel_path.startswith("matching/"):
                        rel_path = rel_path[len("matching/"):]
                    # Old: prepare_dispatch.wgsl, Now: common/prepare_dispatch.wgsl
                    if rel_path.startswith("common/"):
                        # Some common files might be needed by name, e.g. prepare_dispatch
                        # Keep others as is (though C++ might not load them directly)
                        base = os.path.basename(rel_path)
                        if base == "prepare_dispatch.wgsl":
                            rel_path = base
                    
                    content = process_includes(filepath)
                    
                    # Escape quotes and newlines for C++ string literal
                    content_escaped = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n"\n"')
                    
                    f.write(f'        {{ "{rel_path}", "{content_escaped}" }},\n')
        
        f.write("    };\n")
        f.write("    \n")
        f.write("    inline std::string GetShader(const std::string& path) {\n")
        f.write("        auto it = shaders.find(path);\n")
        f.write("        if (it != shaders.end()) return it->second;\n")
        f.write("        return \"\";\n")
        f.write("    }\n")
        f.write("}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python embed_shaders.py <shader_dir> <output_header>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
