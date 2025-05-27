const readline = require("readline").createInterface({
  input: process.stdin,
  output: process.stdout,
});
const vocab = [];

let count = 1;

// Lowercase a–z
for (let i = 97; i <= 122; i++) {
  vocab.push({ char: String.fromCharCode(i), id: count++ });
}

// Uppercase A–Z
for (let i = 65; i <= 90; i++) {
  vocab.push({ char: String.fromCharCode(i), id: count++ });
}

// make object of id and char (vise versa) // example : {a: 1,b: 2,c: 3,d: 4,e: 5}
const charToId = Object.fromEntries(vocab.map(({ char, id }) => [char, id]));
const idToChar = Object.fromEntries(vocab.map(({ char, id }) => [id, char]));

class Token {
  // Encode: Each word becomes a single large number using base-1000
  encode(string) {
    const words = string.split(" ");
    const encoded = words.map((word) => {

      // returns the array of ids for letter in word , example : [ 8, 5, 12, 12, 15 ]
      const ids = word.split("").map((ch) => charToId[ch] || 0);

      // Convert to a single number using base-1000 (large enough for safety)
      let encodedNum = 0;
      ids.forEach((id, idx) => {
        encodedNum += id * Math.pow(1000, idx);
      });

      return encodedNum.toString();
    });

    const final = encoded.join(" "); // or use comma, dot, etc.
    console.log("Encoded:", final);
    return final;
  }

  // Decode: Convert each number back to letters using base-1000
  decode(encodedString) {
    const words = encodedString.split(" ");
    
    const decoded = words.map((numStr) => {

      let num = parseInt(numStr);
      const chars = [];

      while (num > 0) {
        const id = num % 1000;
        chars.push(idToChar[id] || "?");
        num = Math.floor(num / 1000);
      }

      // reconstruct word
      return chars.join(""); 
    });

    const final = decoded.join(" ");
    console.log("Decoded: ", final);
    return final;
  }
}

readline.question("encoding string: ", (string) => {
  const myToken = new Token();
  const encoded = myToken.encode(string);
  myToken.decode(encoded);
  readline.close();
});
