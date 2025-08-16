#!/usr/bin/env python3
"""
Demonstration of what the stream analysis files will look like
"""
import json
import os
from datetime import datetime

def create_demo_stream_file():
    """Create a demo stream analysis file to show the format"""
    
    # Sample reasoning trace data (simplified)
    sample_result = {
        "question": "Tetrataenite is a rare iron-nickel alloy mineral known for its highly ordered atomic structure and impressive magnetic properties. Its formation is closely tied to the slow cooling processes in meteorites. Which of the following characteristics most directly enables tetrataenite to be considered a potential candidate for advanced permanent magnet applications?",
        "options": [
            "Its occurrence exclusively in terrestrial ore deposits",
            "Its lack of magnetic ordering at room temperature", 
            "Its unique ordered atomic arrangement of iron and nickel atoms",
            "Its high solubility in water"
        ],
        "reasoning": {
            "thought_process": {
                "option_1": "This seems unlikely given the formation conditions...",
                "option_2": "This contradicts what we know about magnetic properties...",
                "option_3": "This is the key factor - the ordered structure creates strong magnetic properties...",
                "option_4": "This doesn't make sense for a metallic mineral..."
            },
            "prediction": {
                "predicted_answer": "3",
                "prediction_reasoning": "The ordered atomic arrangement is what gives tetrataenite its magnetic properties"
            }
        }
    }
    
    sample_question = {
        "question": sample_result["question"],
        "options": sample_result["options"]
    }
    
    # Demo stream analysis content
    demo_stream_analysis = """Hmm, let me think about this question carefully. When I read about tetrataenite being a potential candidate for permanent magnets, I need to focus on what property would make it suitable for that application.

Let me consider each possibility that comes to mind...

The first thought that occurs to me is about it occurring exclusively in terrestrial ore deposits. But wait, the question specifically mentions that its formation is tied to slow cooling processes in meteorites. That's actually contradicting the terrestrial origin idea. If it forms in meteorites through slow cooling, then it's not exclusively terrestrial. This doesn't seem right for explaining its magnetic applications anyway.

Then I think about the idea of it lacking magnetic ordering at room temperature. That immediately strikes me as wrong - if we're talking about permanent magnet applications, we absolutely need magnetic ordering at room temperature. That's fundamental to how permanent magnets work. A material that lacks magnetic ordering couldn't function as a permanent magnet, so this can't be the answer.

Now the third possibility really catches my attention - the unique ordered atomic arrangement of iron and nickel atoms. This is making me think about the relationship between crystal structure and magnetic properties. In magnetic materials, the way atoms are arranged in the crystal lattice directly affects the magnetic behavior. Iron and nickel are both ferromagnetic elements, and when they're arranged in a highly ordered structure, this can create very strong magnetic properties. The slow cooling process mentioned in the question would allow atoms to settle into this highly ordered arrangement...

Actually, let me think more about this ordered structure. In permanent magnets, we need materials with high magnetic anisotropy and coercivity. The ordered arrangement of iron and nickel atoms would create preferred magnetic directions and make it harder to demagnetize the material. This is exactly what you want in a permanent magnet material.

The fourth option about high solubility in water - that doesn't make any sense at all. We're talking about a metallic alloy of iron and nickel. These are not water-soluble elements, and being soluble in water would actually be detrimental to any practical application, let alone magnetic applications.

So I keep coming back to that ordered atomic arrangement. That's really the key here. The unique crystal structure with iron and nickel atoms in specific ordered positions is what would give tetrataenite its impressive magnetic properties that make it suitable for permanent magnet applications.

I'm confident that the answer is option 3 - the unique ordered atomic arrangement of iron and nickel atoms."""
    
    # Create output directory
    output_dir = "demo_stream_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create demo file
    filename = "001_Tetrataenite_is_a_rare_iron_nickel_alloy_mineral_k_STREAM_ANALYSIS.txt"
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("🌊 COHERENT STREAM OF THOUGHT ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("QUESTION 1:\n")
        f.write(f"{sample_question['question']}\n\n")
        f.write("STREAM OF THOUGHT:\n")
        f.write("-" * 40 + "\n")
        f.write(demo_stream_analysis)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("Generated by: argo:gpt-4.1\n")
        f.write("Specialty: expert\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Demo stream analysis file created: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    # Show first few lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"\nFirst 10 lines of the file:")
        for i, line in enumerate(lines[:10]):
            print(f"{i+1:2d}: {line.rstrip()}")

if __name__ == "__main__":
    create_demo_stream_file()