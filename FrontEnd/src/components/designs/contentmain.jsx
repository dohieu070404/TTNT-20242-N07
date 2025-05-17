import React, { useState } from "react";
import './contentmain.css';

const ContentMain = () => {
  const [selected, setSelected] = useState(null);

  const providers = [
    { name: "Iacob Geaorgescu", email: "e-mail@test-email.com" },
    { name: "Julius Neumann", email: "e-mail@test-email.com" },
    { name: "Christoph Koller", email: "e-mail@test-email.com" },
    { name: "Bram Lemmens", email: "e-mail@test-email.com" },
  ];

  const handleSelect = (provider) => {
    setSelected(provider);
  };

  return (
    <div className="wrapper">
      <h2>Providers</h2>
      <table className="table">
        <thead className="thead">
          <tr>
            <th className="th">Provider Name</th>
            <th className="th">Email</th>
            <th className="th">Action</th>
          </tr>
        </thead>
        <tbody className="tbody">
          {providers.map((p, i) => (
            <tr key={i} className="tr">
              <td className="td">{p.name}</td>
              <td className="td">{p.email}</td>
              <td className="td">
                <button className="button" onClick={() => handleSelect(p)}>Select</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className={`detail-view ${selected ? 'show' : ''}`}>
        {selected && (
          <>
            <h3>Provider Details</h3>
            <p><strong>Name:</strong> {selected.name}</p>
            <p><strong>Email:</strong> {selected.email}</p>
            <p><strong>City:</strong> Detroit</p>
            <p><strong>Phone:</strong> 555-555-5555</p>
            <p><strong>Last Update:</strong> Jun 20 2014</p>
            <p><strong>Notes:</strong> Lorem ipsum dolor sit amet...</p>
            <button className="button" onClick={() => setSelected(null)}>Close</button>
          </>
        )}
      </div>
    </div>
  );
};

export default ContentMain;
