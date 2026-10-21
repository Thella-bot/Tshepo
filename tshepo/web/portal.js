/* Tshepo capture portal.
 *
 * The whole flow runs in the applicant's browser: read the file, post it,
 * read the passport, delete the data. No app to install, and the export file
 * is never handed to an email provider, a messaging app or an agent.
 */
(function () {
  "use strict";

  var STORE_KEY = "tshepo.session";
  var session = null;

  function $(id) { return document.getElementById(id); }

  function setStatus(el, text, kind) {
    el.textContent = text;
    el.className = kind ? (kind === "ok" ? "muted status-ok" : "muted status-bad") : "muted";
  }

  function newUserId() {
    if (window.crypto && window.crypto.randomUUID) { return "u_" + window.crypto.randomUUID(); }
    return "u_" + Math.random().toString(36).slice(2) + Date.now().toString(36);
  }

  function loadSession() {
    try { session = JSON.parse(window.localStorage.getItem(STORE_KEY) || "null"); } catch (e) { session = null; }
    return session;
  }

  function saveSession(s) {
    session = s;
    try { window.localStorage.setItem(STORE_KEY, JSON.stringify(s)); } catch (e) { /* private mode */ }
  }

  function clearSession() {
    session = null;
    try { window.localStorage.removeItem(STORE_KEY); } catch (e) { /* ignore */ }
  }

  function api(path, options) {
    options = options || {};
    var headers = { "Content-Type": "application/json" };
    if (session && session.consent_id) {
      headers["X-Consent-Id"] = session.consent_id;
      headers["X-Consent-Token"] = session.consent_token;
    }
    return fetch(path, {
      method: options.method || "GET",
      headers: headers,
      body: options.body ? JSON.stringify(options.body) : undefined
    }).then(function (res) {
      return res.text().then(function (text) {
        var data = {};
        try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { detail: text }; }
        if (!res.ok) {
          var err = new Error(data.detail || ("request failed: " + res.status));
          err.status = res.status;
          throw err;
        }
        return data;
      });
    });
  }

  function showNotice(summary) {
    var box = $("notice");
    box.textContent = summary.text || "";
    if (summary.text_sesotho_draft) {
      var sn = document.createElement("div");
      sn.style.marginTop = "0.8rem";
      sn.style.paddingTop = "0.8rem";
      sn.style.borderTop = "1px solid #d9e0db";
      sn.textContent = summary.text_sesotho_draft;
      box.appendChild(sn);
    }
  }

  function renderPassport(p) {
    var box = $("passport");
    box.innerHTML = "";

    var head = document.createElement("div");
    if (p.score === null || p.score === undefined) {
      var none = document.createElement("div");
      none.className = "score";
      none.textContent = "--";
      head.appendChild(none);
      var nb = document.createElement("p");
      nb.className = "band";
      nb.textContent = "No score yet";
      head.appendChild(nb);
    } else {
      var sc = document.createElement("div");
      sc.className = "score";
      sc.textContent = p.score;
      head.appendChild(sc);
      var band = document.createElement("p");
      band.className = "band";
      band.textContent = p.band;
      head.appendChild(band);
    }
    box.appendChild(head);

    var issues = (p.data_quality && p.data_quality.issues) || [];
    if (issues.length) {
      var box2 = document.createElement("div");
      box2.className = "issues";
      var strong = document.createElement("strong");
      strong.textContent = "We could not make a decision from these messages:";
      box2.appendChild(strong);
      var ul = document.createElement("ul");
      issues.forEach(function (i) {
        var li = document.createElement("li");
        li.textContent = i;
        ul.appendChild(li);
      });
      box2.appendChild(ul);
      box.appendChild(box2);
    }

    var rows = [
      ["Status", p.status],
      ["Transactions used", p.transactions_used],
      ["History (days)", p.history_days],
      ["Weeks with income", p.income_active_weeks],
      ["Monthly income estimate", p.monthly_income_estimate],
      ["Average balance", p.avg_balance],
      ["Loan repayments seen", p.loan_repayments_seen],
      ["Balance chain", p.data_integrity]
    ];
    var table = document.createElement("table");
    rows.forEach(function (r) {
      if (r[1] === undefined || r[1] === null) { return; }
      var tr = document.createElement("tr");
      var th = document.createElement("th");
      th.textContent = r[0];
      var td = document.createElement("td");
      td.textContent = r[1];
      tr.appendChild(th);
      tr.appendChild(td);
      table.appendChild(tr);
    });
    box.appendChild(table);

    if (p.reasons_down && p.reasons_down.length) {
      var ul2 = document.createElement("ul");
      p.reasons_down.forEach(function (r) {
        var li = document.createElement("li");
        li.textContent = r;
        ul2.appendChild(li);
      });
      box.appendChild(ul2);
    }
  }

  function restore() {
    if (!session) { return false; }
    $("user-code").textContent = session.user_id;
    $("step-consent").hidden = true;
    $("step-send").hidden = false;
    $("step-score").hidden = false;
    $("step-delete").hidden = false;
    return true;
  }

  document.addEventListener("DOMContentLoaded", function () {
    api("/v1/consent").then(function (summary) { showNotice(summary); })
      .catch(function (e) { $("notice").textContent = "Could not load the notice: " + e.message; });

    var existing = loadSession();
    if (existing && existing.user_id) {
      $("user-code").textContent = existing.user_id;
    } else {
      $("user-code").textContent = newUserId();
    }

    $("agree").addEventListener("change", function () {
      $("btn-consent").disabled = !$("agree").checked;
    });

    $("btn-consent").addEventListener("click", function () {
      var btn = $("btn-consent");
      btn.disabled = true;
      setStatus($("consent-status"), "Recording your consent...");
      api("/v1/consents", {
        method: "POST",
        body: { user_id: $("user-code").textContent, source: "portal", purpose: "credit_assessment" }
      }).then(function (data) {
        saveSession({ user_id: data.user_id, consent_id: data.consent_id, consent_token: data.consent_token });
        setStatus($("consent-status"), "Consent recorded. It expires " +
          String(data.expires_at).slice(0, 10) + ".", "ok");
        restore();
      }).catch(function (e) {
        btn.disabled = false;
        setStatus($("consent-status"), e.message, "bad");
      });
    });

    $("file").addEventListener("change", function () {
      $("btn-send").disabled = !$("file").files.length;
    });

    $("btn-send").addEventListener("click", function () {
      var file = $("file").files[0];
      if (!file) { return; }
      if (file.size > 8 * 1024 * 1024) {
        setStatus($("send-status"), "That file is too large (8 MB limit).", "bad");
        return;
      }
      setStatus($("send-status"), "Reading your file...");
      var reader = new FileReader();
      reader.onload = function () {
        setStatus($("send-status"), "Sending...");
        api("/v1/ingest", {
          method: "POST",
          body: { consent_id: session.consent_id, user_id: session.user_id, export_text: String(reader.result) }
        }).then(function (data) {
          setStatus($("send-status"), "Received " + data.messages_received + " messages. " +
            "Now build your passport.", "ok");
          $("btn-score").disabled = false;
        }).catch(function (e) {
          setStatus($("send-status"), e.message, "bad");
        });
      };
      reader.onerror = function () { setStatus($("send-status"), "Could not read that file.", "bad"); };
      reader.readAsText(file);
    });

    $("btn-score").addEventListener("click", function () {
      setStatus($("send-status"), "Building your passport...");
      api("/v1/score?consent_id=" + encodeURIComponent(session.consent_id), { method: "POST" })
        .then(function (p) {
          renderPassport(p);
          setStatus($("send-status"), "Done.", "ok");
        }).catch(function (e) {
          setStatus($("send-status"), e.message, "bad");
        });
    });

    $("btn-delete").addEventListener("click", function () {
      if (!window.confirm("Delete every message and your passport?")) { return; }
      api("/v1/users/" + encodeURIComponent(session.user_id) + "/data", { method: "DELETE" })
        .then(function (d) {
          setStatus($("delete-status"),
            "Deleted " + d.messages + " message(s) and " + d.passports + " passport(s).", "ok");
          $("passport").innerHTML = "";
        }).catch(function (e) {
          setStatus($("delete-status"), e.message, "bad");
        });
    });
  });
})();
